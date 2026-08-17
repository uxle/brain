//! Core Tensor struct implementation for the Brain deep learning framework.
//!
//! This module defines the [`Tensor`] struct and all fundamental constructors, accessors,
//! memory representations, shape transformations, splitting/chunking routines, device/dtype
//! conversions, operator overloads, and mutation primitives.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{BrainError, BrainResult};
use crate::random::{self, BrainRng, Rng};

// =============================================================================
// Helper Functions for Strides and Layout
// =============================================================================

/// Computes standard row-major (C-order) strides from a shape slice.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// =============================================================================
// Tensor Struct Definition
// =============================================================================

/// A multi-dimensional array of numerical data.
///
/// The Tensor is the central data structure of the Brain framework, representing
/// an n-dimensional array with a given shape, data type, device placement,
/// and optional gradient tracking.
///
/// Internally, data is stored as a contiguous `Vec<f64>` in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    device: Device,
    dtype: DType,
    requires_grad: bool,
    name: Option<String>,
}

// =============================================================================
// Tensor Implementation - Constructors
// =============================================================================

impl Tensor {
    /// Creates a new tensor from a data vector and shape.
    ///
    /// # Panics
    ///
    /// Panics if the product of shape dimensions does not equal `data.len()`.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} does not match shape product {}",
            data.len(),
            numel
        );
        let strides = compute_strides(&shape);
        Tensor {
            data,
            shape,
            strides,
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a tensor from an owned vector and shape.
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self::new(data, shape)
    }

    /// Creates a tensor from a slice and shape.
    pub fn from_slice(data: &[f64], shape: Vec<usize>) -> Self {
        Self::new(data.to_vec(), shape)
    }

    /// Creates a scalar (0-dimensional) tensor.
    pub fn scalar(value: f64) -> Self {
        Tensor {
            data: vec![value],
            shape: vec![],
            strides: vec![],
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; numel],
            shape: shape.clone(),
            strides: compute_strides(&shape),
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            data: vec![1.0; numel],
            shape: shape.clone(),
            strides: compute_strides(&shape),
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: f64) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            data: vec![value; numel],
            shape: shape.clone(),
            strides: compute_strides(&shape),
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a 1D tensor with values linearly spaced in `[start, end)`.
    ///
    /// Supports both positive and negative steps. With `step > 0` the sequence
    /// runs from `start` upward while `< end`; with `step < 0` it runs downward
    /// while `> end`. A zero step panics.
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        assert!(step != 0.0, "arange step must be non-zero");
        let mut data = Vec::new();
        if step > 0.0 {
            let mut current = start;
            while current < end {
                data.push(current);
                current += step;
            }
        } else {
            let mut current = start;
            while current > end {
                data.push(current);
                current += step;
            }
        }
        let len = data.len();
        Self::new(data, vec![len])
    }

    /// Creates a 1D tensor with `steps` values evenly spaced in `[start, end]`.
    pub fn linspace(start: f64, end: f64, steps: usize) -> Self {
        assert!(steps >= 1, "linspace steps must be >= 1");
        if steps == 1 {
            return Self::new(vec![start], vec![1]);
        }
        let step = (end - start) / ((steps - 1) as f64);
        let mut data = Vec::with_capacity(steps);
        for i in 0..steps {
            data.push(start + (i as f64) * step);
        }
        Self::new(data, vec![steps])
    }

    /// Creates a 2D identity matrix of size `n x n`.
    pub fn eye(n: usize) -> Self {
        let mut t = Self::zeros(vec![n, n]);
        for i in 0..n {
            t.set_2d(i, i, 1.0);
        }
        t
    }

    /// Creates a 2D identity matrix of size `n x n` (alias).
    pub fn identity(n: usize) -> Self {
        Self::eye(n)
    }

    /// Creates an empty tensor with zero elements.
    pub fn empty(shape: Vec<usize>) -> Self {
        Self::new(Vec::new(), shape)
    }

    /// Creates a tensor filled with standard uniform random values in `[0, 1)`.
    pub fn rand(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let mut data = vec![0.0; numel];
        random::with_rng(|rng| {
            rng.fill_f64_slice(&mut data);
        });
        Self::new(data, shape)
    }

    /// Creates a tensor filled with standard normal random values (mean 0, std 1).
    pub fn randn(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        random::with_rng(|rng| {
            for _ in 0..numel {
                let u1 = (1.0 - rng.next_f64()).max(1e-15);
                let u2 = rng.next_f64();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                data.push(z0);
            }
        });
        Self::new(data, shape)
    }
}

// =============================================================================
// Tensor Implementation - Accessors & Metadata
// =============================================================================

impl Tensor {
    /// Returns an immutable slice to the underlying data buffer.
    #[inline(always)]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable slice to the underlying data buffer.
    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns the shape slice.
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides slice.
    #[inline(always)]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the number of dimensions (rank).
    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    #[inline(always)]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the tensor has 0 elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the device placement of this tensor.
    #[inline(always)]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the data type of this tensor.
    #[inline(always)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns whether gradient computation is required.
    #[inline(always)]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Sets whether gradient computation is required.
    pub fn set_requires_grad(&mut self, req: bool) {
        self.requires_grad = req;
    }

    /// Returns the optional debug name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Sets the optional debug name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Gets an element at a flat index.
    #[inline(always)]
    pub fn get(&self, flat_index: usize) -> f64 {
        self.data[flat_index]
    }

    /// Sets an element at a flat index.
    #[inline(always)]
    pub fn set(&mut self, flat_index: usize, value: f64) {
        self.data[flat_index] = value;
    }

    /// Computes the flat row-major index for multi-dimensional coordinates.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Coordinate rank {} does not match tensor rank {}",
            indices.len(),
            self.shape.len()
        );
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx < self.shape[i],
                "Index {} out of bounds for axis {} of size {}",
                idx,
                i,
                self.shape[i]
            );
            offset += idx * self.strides[i];
        }
        offset
    }

    /// Gets an element at multi-dimensional coordinates.
    pub fn get_index(&self, indices: &[usize]) -> f64 {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    /// Sets an element at multi-dimensional coordinates.
    pub fn set_index(&mut self, indices: &[usize], value: f64) {
        let idx = self.flat_index(indices);
        self.data[idx] = value;
    }

    /// Gets an element from a 2D matrix.
    #[inline(always)]
    pub fn get_2d(&self, r: usize, c: usize) -> f64 {
        assert_eq!(self.shape.len(), 2, "get_2d requires a 2D tensor");
        assert!(r < self.shape[0] && c < self.shape[1], "2D index out of bounds");
        self.data[r * self.strides[0] + c * self.strides[1]]
    }

    /// Sets an element in a 2D matrix.
    #[inline(always)]
    pub fn set_2d(&mut self, r: usize, c: usize, value: f64) {
        assert_eq!(self.shape.len(), 2, "set_2d requires a 2D tensor");
        assert!(r < self.shape[0] && c < self.shape[1], "2D index out of bounds");
        let idx = r * self.strides[0] + c * self.strides[1];
        self.data[idx] = value;
    }

    /// Gets an element from a 3D tensor.
    #[inline(always)]
    pub fn get_3d(&self, d0: usize, d1: usize, d2: usize) -> f64 {
        assert_eq!(self.shape.len(), 3, "get_3d requires a 3D tensor");
        let idx = d0 * self.strides[0] + d1 * self.strides[1] + d2 * self.strides[2];
        self.data[idx]
    }

    /// Sets an element in a 3D tensor.
    #[inline(always)]
    pub fn set_3d(&mut self, d0: usize, d1: usize, d2: usize, value: f64) {
        assert_eq!(self.shape.len(), 3, "set_3d requires a 3D tensor");
        let idx = d0 * self.strides[0] + d1 * self.strides[1] + d2 * self.strides[2];
        self.data[idx] = value;
    }

    /// Gets an element from a 4D tensor.
    #[inline(always)]
    pub fn get_4d(&self, d0: usize, d1: usize, d2: usize, d3: usize) -> f64 {
        assert_eq!(self.shape.len(), 4, "get_4d requires a 4D tensor");
        let idx = d0 * self.strides[0]
            + d1 * self.strides[1]
            + d2 * self.strides[2]
            + d3 * self.strides[3];
        self.data[idx]
    }

    /// Sets an element in a 4D tensor.
    #[inline(always)]
    pub fn set_4d(&mut self, d0: usize, d1: usize, d2: usize, d3: usize, value: f64) {
        assert_eq!(self.shape.len(), 4, "set_4d requires a 4D tensor");
        let idx = d0 * self.strides[0]
            + d1 * self.strides[1]
            + d2 * self.strides[2]
            + d3 * self.strides[3];
        self.data[idx] = value;
    }
}

// =============================================================================
// Tensor Conversions & Transforms
// =============================================================================

impl Tensor {
    /// Copies tensor data into a flat `Vec<f64>`.
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Converts a 2D tensor into nested `Vec<Vec<f64>>`.
    pub fn to_vec_2d(&self) -> Vec<Vec<f64>> {
        assert_eq!(self.ndim(), 2, "to_vec_2d requires a 2D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(self.get_2d(r, c));
            }
            result.push(row);
        }
        result
    }

    /// Converts a 3D tensor into nested `Vec<Vec<Vec<f64>>>`.
    pub fn to_vec_3d(&self) -> Vec<Vec<Vec<f64>>> {
        assert_eq!(self.ndim(), 3, "to_vec_3d requires a 3D tensor");
        let (d0, d1, d2) = (self.shape[0], self.shape[1], self.shape[2]);
        let mut result = Vec::with_capacity(d0);
        for i in 0..d0 {
            let mut matrix = Vec::with_capacity(d1);
            for j in 0..d1 {
                let mut row = Vec::with_capacity(d2);
                for k in 0..d2 {
                    row.push(self.get_3d(i, j, k));
                }
                matrix.push(row);
            }
            result.push(matrix);
        }
        result
    }

    /// Moves the tensor to a target device (returns cloned copy with updated device metadata).
    pub fn to_device(&self, device: Device) -> Self {
        let mut t = self.clone();
        t.device = device;
        t
    }

    /// Moves the tensor to the host CPU.
    pub fn cpu(&self) -> Self {
        self.to_device(Device::Cpu)
    }

    /// Casts data type (returns a tensor with new dtype tag).
    pub fn dtype_cast(&self, dtype: DType) -> Self {
        let mut t = self.clone();
        t.dtype = dtype;
        t
    }

    /// Sets the dtype in place.
    pub fn set_dtype(&mut self, dtype: DType) {
        self.dtype = dtype;
    }

    /// Checks whether memory layout is contiguous row-major.
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape);
        self.strides == expected
    }

    /// Returns a contiguous copy of this tensor.
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            let numel = self.numel();
            let mut data = Vec::with_capacity(numel);
            let mut coords = vec![0usize; self.ndim()];
            for _ in 0..numel {
                data.push(self.get_index(&coords));
                // Increment coordinates
                for dim in (0..self.ndim()).rev() {
                    coords[dim] += 1;
                    if coords[dim] < self.shape[dim] {
                        break;
                    }
                    coords[dim] = 0;
                }
            }
            Self::new(data, self.shape.clone())
        }
    }

    /// Makes the tensor contiguous in place.
    pub fn make_contiguous(&mut self) {
        if !self.is_contiguous() {
            *self = self.contiguous();
        }
    }

    /// Reshapes the tensor to a new shape with the same number of elements.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Reshape numel {} does not match source numel {}",
            new_numel,
            self.numel()
        );
        let contiguous_t = self.contiguous();
        Self::new(contiguous_t.data, new_shape)
    }

    /// Creates a view with a new shape if memory layout is contiguous.
    pub fn view(&self, new_shape: Vec<usize>) -> BrainResult<Self> {
        let new_numel: usize = new_shape.iter().product();
        if self.numel() != new_numel {
            return Err(BrainError::shape_mismatch(
                format!("numel {}", new_numel),
                format!("numel {}", self.numel()),
                "view: shape product mismatch",
            ));
        }
        if !self.is_contiguous() {
            return Err(BrainError::invalid_value(
                "view size is not compatible with input tensor's size and stride (use contiguous())",
            ));
        }
        Ok(Self::new(self.data.clone(), new_shape))
    }

    /// Permutes dimensions according to a given permutation.
    pub fn permute(&self, permutation: &[usize]) -> Self {
        assert_eq!(
            permutation.len(),
            self.ndim(),
            "Permutation length must equal tensor rank"
        );
        let mut new_shape = vec![0; self.ndim()];
        let mut new_strides = vec![0; self.ndim()];
        for (i, &p) in permutation.iter().enumerate() {
            new_shape[i] = self.shape[p];
            new_strides[i] = self.strides[p];
        }
        let mut t = Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            device: self.device,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            name: self.name.clone(),
        };
        t.make_contiguous();
        t
    }

    /// Transposes two dimensions of the tensor.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(dim0 < self.ndim() && dim1 < self.ndim(), "Transpose dims out of range");
        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        perm.swap(dim0, dim1);
        self.permute(&perm)
    }

    /// Transposes a 2D matrix (shorthand for `.transpose(0, 1)`).
    pub fn t(&self) -> Self {
        assert_eq!(self.ndim(), 2, "t() requires a 2D tensor");
        self.transpose(0, 1)
    }

    /// Removes all dimensions of size 1.
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().copied().filter(|&d| d != 1).collect();
        self.reshape(new_shape)
    }

    /// Removes a specific dimension of size 1.
    pub fn squeeze_dim(&self, dim: usize) -> Self {
        assert!(dim < self.ndim(), "squeeze_dim: dim out of bounds");
        if self.shape[dim] == 1 {
            let mut new_shape = self.shape.clone();
            new_shape.remove(dim);
            self.reshape(new_shape)
        } else {
            self.clone()
        }
    }

    /// Inserts a dimension of size 1 at index `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Self {
        assert!(dim <= self.ndim(), "unsqueeze: dim out of bounds");
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(new_shape)
    }

    /// Flattens dimensions from `start_dim` to `end_dim` inclusive into a single dimension.
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Self {
        assert!(start_dim <= end_dim && end_dim < self.ndim(), "flatten: invalid dim range");
        let mut new_shape = Vec::new();
        for i in 0..start_dim {
            new_shape.push(self.shape[i]);
        }
        let flattened_dim: usize = self.shape[start_dim..=end_dim].iter().product();
        new_shape.push(flattened_dim);
        for i in end_dim + 1..self.ndim() {
            new_shape.push(self.shape[i]);
        }
        self.reshape(new_shape)
    }

    /// Expands a dimension into multiple dimensions.
    pub fn unflatten(&self, dim: usize, sizes: &[usize]) -> Self {
        assert!(dim < self.ndim(), "unflatten: dim out of bounds");
        let prod: usize = sizes.iter().product();
        assert_eq!(
            self.shape[dim], prod,
            "unflatten: product of sizes {} must match dim size {}",
            prod, self.shape[dim]
        );
        let mut new_shape = Vec::new();
        for i in 0..dim {
            new_shape.push(self.shape[i]);
        }
        new_shape.extend_from_slice(sizes);
        for i in dim + 1..self.ndim() {
            new_shape.push(self.shape[i]);
        }
        self.reshape(new_shape)
    }

    /// Expands singleton dimensions to match the given target shape.
    pub fn expand(&self, target_shape: &[usize]) -> BrainResult<Self> {
        let src_rank = self.ndim();
        let tgt_rank = target_shape.len();
        if src_rank > tgt_rank {
            return Err(BrainError::shape_mismatch(
                format!("{:?}", target_shape),
                format!("{:?}", self.shape),
                "expand: target rank cannot be smaller than source rank",
            ));
        }

        let mut out_data = Vec::new();
        let target_numel: usize = target_shape.iter().product();
        let mut coords = vec![0usize; tgt_rank];

        for _ in 0..target_numel {
            let mut src_coords = Vec::with_capacity(src_rank);
            for i in 0..src_rank {
                let tgt_idx = coords[tgt_rank - src_rank + i];
                let src_dim = self.shape[i];
                if src_dim == 1 {
                    src_coords.push(0);
                } else if src_dim == target_shape[tgt_rank - src_rank + i] {
                    src_coords.push(tgt_idx);
                } else {
                    return Err(BrainError::shape_mismatch(
                        format!("{:?}", target_shape),
                        format!("{:?}", self.shape),
                        "expand: incompatible dimensions",
                    ));
                }
            }
            out_data.push(self.get_index(&src_coords));

            // Increment coords
            for dim in (0..tgt_rank).rev() {
                coords[dim] += 1;
                if coords[dim] < target_shape[dim] {
                    break;
                }
                coords[dim] = 0;
            }
        }

        Ok(Self::new(out_data, target_shape.to_vec()))
    }

    /// Expands this tensor to match the shape of another tensor.
    pub fn expand_as(&self, other: &Tensor) -> BrainResult<Self> {
        self.expand(other.shape())
    }
}

// =============================================================================
// Splitting and Chunking
// =============================================================================

impl Tensor {
    /// Splits the tensor into chunks of `split_size` along dimension `dim`.
    pub fn split(&self, split_size: usize, dim: usize) -> Vec<Tensor> {
        assert!(dim < self.ndim(), "split: dim out of bounds");
        assert!(split_size > 0, "split_size must be positive");
        let dim_len = self.shape[dim];
        let mut results = Vec::new();
        let mut start = 0;
        while start < dim_len {
            let len = (dim_len - start).min(split_size);
            results.push(self.narrow(dim, start, len));
            start += len;
        }
        results
    }

    /// Splits the tensor into a specific number of equal or near-equal chunks along `dim`.
    pub fn chunk(&self, chunks: usize, dim: usize) -> Vec<Tensor> {
        assert!(chunks > 0, "chunks must be > 0");
        assert!(dim < self.ndim(), "chunk: dim out of bounds");
        let dim_len = self.shape[dim];
        let split_size = (dim_len + chunks - 1) / chunks;
        self.split(split_size, dim)
    }

    /// Narrows a dimension to a sub-range `[start, start + length)`.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Self {
        assert!(dim < self.ndim(), "narrow: dim out of bounds");
        assert!(
            start + length <= self.shape[dim],
            "narrow: start + length exceeds dimension size"
        );
        let mut new_shape = self.shape.clone();
        new_shape[dim] = length;

        let numel: usize = new_shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut coords = vec![0usize; self.ndim()];

        for _ in 0..numel {
            let mut src_coords = coords.clone();
            src_coords[dim] += start;
            data.push(self.get_index(&src_coords));

            for d in (0..self.ndim()).rev() {
                coords[d] += 1;
                if coords[d] < new_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
        Self::new(data, new_shape)
    }

    /// Selects an index along a dimension, returning a tensor with rank reduced by 1.
    pub fn select(&self, dim: usize, index: usize) -> Self {
        assert!(dim < self.ndim(), "select: dim out of bounds");
        assert!(index < self.shape[dim], "select: index out of bounds");
        let narrowed = self.narrow(dim, index, 1);
        narrowed.squeeze_dim(dim)
    }

    /// Slices along a dimension with `start`, `end`, and `step`.
    pub fn slice(&self, dim: usize, start: usize, end: usize, step: usize) -> Self {
        assert!(dim < self.ndim(), "slice: dim out of bounds");
        assert!(step > 0, "slice: step must be > 0");
        let actual_end = end.min(self.shape[dim]);
        let mut slice_indices = Vec::new();
        let mut cur = start;
        while cur < actual_end {
            slice_indices.push(cur);
            cur += step;
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = slice_indices.len();

        let numel: usize = new_shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut coords = vec![0usize; self.ndim()];

        for _ in 0..numel {
            let mut src_coords = coords.clone();
            src_coords[dim] = slice_indices[coords[dim]];
            data.push(self.get_index(&src_coords));

            for d in (0..self.ndim()).rev() {
                coords[d] += 1;
                if coords[d] < new_shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }
        Self::new(data, new_shape)
    }

    /// Maps a function element-wise over the tensor.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        Self::new(data, self.shape.clone())
    }

    /// Maps a binary function over two tensors with broadcasting.
    pub fn map2<F>(&self, other: &Tensor, f: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        let common_shape = crate::shape::Shape::broadcast_shapes(&[
            &crate::shape::Shape::from_dims(self.shape()),
            &crate::shape::Shape::from_dims(other.shape()),
        ])
        .expect("Tensors cannot be broadcast together");

        let a_exp = self.expand(common_shape.as_slice()).unwrap();
        let b_exp = other.expand(common_shape.as_slice()).unwrap();

        let data: Vec<f64> = a_exp
            .data()
            .iter()
            .zip(b_exp.data().iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        Self::new(data, common_shape.to_vec())
    }
}

// =============================================================================
// Operator Overloads
// =============================================================================

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a + b)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        &self + &rhs
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a - b)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        &self - &rhs
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a * b)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        &self * &rhs
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a / b)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        &self / &rhs
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.map(|x| -x)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        -&self
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        self.data == other.data
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tensor(shape={:?}, dtype={:?}):", self.shape, self.dtype)?;
        if self.ndim() <= 2 {
            let rows = if self.ndim() == 2 { self.shape[0] } else { 1 };
            let cols = if self.ndim() == 2 { self.shape[1] } else { self.numel() };
            for r in 0..rows {
                write!(f, "  [")?;
                for c in 0..cols {
                    let idx = if self.ndim() == 2 { r * self.strides[0] + c * self.strides[1] } else { c };
                    write!(f, " {:8.4}", self.data[idx])?;
                }
                writeln!(f, " ]")?;
            }
        } else {
            writeln!(f, "  [ ... {} elements ... ]", self.numel())?;
        }
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_constructors() {
        let z = Tensor::zeros(vec![2, 3]);
        assert_eq!(z.numel(), 6);
        assert_eq!(z.get(0), 0.0);

        let o = Tensor::ones(vec![4]);
        assert_eq!(o.numel(), 4);
        assert_eq!(o.get(3), 1.0);

        let f = Tensor::full(vec![2, 2], 5.5);
        assert_eq!(f.get_2d(1, 1), 5.5);

        let eye = Tensor::eye(3);
        assert_eq!(eye.get_2d(0, 0), 1.0);
        assert_eq!(eye.get_2d(0, 1), 0.0);
        assert_eq!(eye.get_2d(1, 1), 1.0);
    }

    #[test]
    fn test_arange_and_linspace() {
        let a = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);

        let l = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(l.shape(), &[5]);
        assert_eq!(l.data(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_tensor_reshape_and_view() {
        let t = Tensor::arange(0.0, 6.0, 1.0);
        let r = t.reshape(vec![2, 3]);
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.get_2d(1, 0), 3.0);

        let v = t.view(vec![3, 2]).unwrap();
        assert_eq!(v.shape(), &[3, 2]);
    }

    #[test]
    fn test_tensor_transpose_and_permute() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tr = t.t();
        assert_eq!(tr.shape(), &[3, 2]);
        assert_eq!(tr.get_2d(0, 1), 4.0);
        assert_eq!(tr.get_2d(2, 0), 3.0);
    }

    #[test]
    fn test_tensor_squeeze_unsqueeze() {
        let t = Tensor::zeros(vec![1, 3, 1, 4]);
        let s = t.squeeze();
        assert_eq!(s.shape(), &[3, 4]);

        let u = s.unsqueeze(0);
        assert_eq!(u.shape(), &[1, 3, 4]);
    }

    #[test]
    fn test_tensor_flatten_unflatten() {
        let t = Tensor::zeros(vec![2, 3, 4, 5]);
        let f = t.flatten(1, 2);
        assert_eq!(f.shape(), &[2, 12, 5]);

        let uf = f.unflatten(1, &[3, 4]);
        assert_eq!(uf.shape(), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_tensor_split_and_chunk() {
        let t = Tensor::arange(0.0, 10.0, 1.0);
        let chunks = t.chunk(3, 0);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].numel(), 4);
        assert_eq!(chunks[1].numel(), 4);
        assert_eq!(chunks[2].numel(), 2);
    }

    #[test]
    fn test_tensor_operators() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let c = &a + &b;
        assert_eq!(c.data(), &[4.0, 6.0]);

        let d = &a * &b;
        assert_eq!(d.data(), &[3.0, 8.0]);

        let neg = -&a;
        assert_eq!(neg.data(), &[-1.0, -2.0]);
    }

    #[test]
    fn test_tensor_stress_case_001() {
        let n = (1 % 16) + 1;
        let t = Tensor::linspace(0.0, 1.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_002() {
        let n = (2 % 16) + 1;
        let t = Tensor::linspace(0.0, 2.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_003() {
        let n = (3 % 16) + 1;
        let t = Tensor::linspace(0.0, 3.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_004() {
        let n = (4 % 16) + 1;
        let t = Tensor::linspace(0.0, 4.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_005() {
        let n = (5 % 16) + 1;
        let t = Tensor::linspace(0.0, 5.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_006() {
        let n = (6 % 16) + 1;
        let t = Tensor::linspace(0.0, 6.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_007() {
        let n = (7 % 16) + 1;
        let t = Tensor::linspace(0.0, 7.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_008() {
        let n = (8 % 16) + 1;
        let t = Tensor::linspace(0.0, 8.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_009() {
        let n = (9 % 16) + 1;
        let t = Tensor::linspace(0.0, 9.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_010() {
        let n = (10 % 16) + 1;
        let t = Tensor::linspace(0.0, 10.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_011() {
        let n = (11 % 16) + 1;
        let t = Tensor::linspace(0.0, 11.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_012() {
        let n = (12 % 16) + 1;
        let t = Tensor::linspace(0.0, 12.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_013() {
        let n = (13 % 16) + 1;
        let t = Tensor::linspace(0.0, 13.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_014() {
        let n = (14 % 16) + 1;
        let t = Tensor::linspace(0.0, 14.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_015() {
        let n = (15 % 16) + 1;
        let t = Tensor::linspace(0.0, 15.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_016() {
        let n = (16 % 16) + 1;
        let t = Tensor::linspace(0.0, 16.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_017() {
        let n = (17 % 16) + 1;
        let t = Tensor::linspace(0.0, 17.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_018() {
        let n = (18 % 16) + 1;
        let t = Tensor::linspace(0.0, 18.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_019() {
        let n = (19 % 16) + 1;
        let t = Tensor::linspace(0.0, 19.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_020() {
        let n = (20 % 16) + 1;
        let t = Tensor::linspace(0.0, 20.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_021() {
        let n = (21 % 16) + 1;
        let t = Tensor::linspace(0.0, 21.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_022() {
        let n = (22 % 16) + 1;
        let t = Tensor::linspace(0.0, 22.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_023() {
        let n = (23 % 16) + 1;
        let t = Tensor::linspace(0.0, 23.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_024() {
        let n = (24 % 16) + 1;
        let t = Tensor::linspace(0.0, 24.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_025() {
        let n = (25 % 16) + 1;
        let t = Tensor::linspace(0.0, 25.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_026() {
        let n = (26 % 16) + 1;
        let t = Tensor::linspace(0.0, 26.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_027() {
        let n = (27 % 16) + 1;
        let t = Tensor::linspace(0.0, 27.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_028() {
        let n = (28 % 16) + 1;
        let t = Tensor::linspace(0.0, 28.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_029() {
        let n = (29 % 16) + 1;
        let t = Tensor::linspace(0.0, 29.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_030() {
        let n = (30 % 16) + 1;
        let t = Tensor::linspace(0.0, 30.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_031() {
        let n = (31 % 16) + 1;
        let t = Tensor::linspace(0.0, 31.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_032() {
        let n = (32 % 16) + 1;
        let t = Tensor::linspace(0.0, 32.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_033() {
        let n = (33 % 16) + 1;
        let t = Tensor::linspace(0.0, 33.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_034() {
        let n = (34 % 16) + 1;
        let t = Tensor::linspace(0.0, 34.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_035() {
        let n = (35 % 16) + 1;
        let t = Tensor::linspace(0.0, 35.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_036() {
        let n = (36 % 16) + 1;
        let t = Tensor::linspace(0.0, 36.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_037() {
        let n = (37 % 16) + 1;
        let t = Tensor::linspace(0.0, 37.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_038() {
        let n = (38 % 16) + 1;
        let t = Tensor::linspace(0.0, 38.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_039() {
        let n = (39 % 16) + 1;
        let t = Tensor::linspace(0.0, 39.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_040() {
        let n = (40 % 16) + 1;
        let t = Tensor::linspace(0.0, 40.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_041() {
        let n = (41 % 16) + 1;
        let t = Tensor::linspace(0.0, 41.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_042() {
        let n = (42 % 16) + 1;
        let t = Tensor::linspace(0.0, 42.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_043() {
        let n = (43 % 16) + 1;
        let t = Tensor::linspace(0.0, 43.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_044() {
        let n = (44 % 16) + 1;
        let t = Tensor::linspace(0.0, 44.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_045() {
        let n = (45 % 16) + 1;
        let t = Tensor::linspace(0.0, 45.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_046() {
        let n = (46 % 16) + 1;
        let t = Tensor::linspace(0.0, 46.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_047() {
        let n = (47 % 16) + 1;
        let t = Tensor::linspace(0.0, 47.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_048() {
        let n = (48 % 16) + 1;
        let t = Tensor::linspace(0.0, 48.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_049() {
        let n = (49 % 16) + 1;
        let t = Tensor::linspace(0.0, 49.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_050() {
        let n = (50 % 16) + 1;
        let t = Tensor::linspace(0.0, 50.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_051() {
        let n = (51 % 16) + 1;
        let t = Tensor::linspace(0.0, 51.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_052() {
        let n = (52 % 16) + 1;
        let t = Tensor::linspace(0.0, 52.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_053() {
        let n = (53 % 16) + 1;
        let t = Tensor::linspace(0.0, 53.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_054() {
        let n = (54 % 16) + 1;
        let t = Tensor::linspace(0.0, 54.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_055() {
        let n = (55 % 16) + 1;
        let t = Tensor::linspace(0.0, 55.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_056() {
        let n = (56 % 16) + 1;
        let t = Tensor::linspace(0.0, 56.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_057() {
        let n = (57 % 16) + 1;
        let t = Tensor::linspace(0.0, 57.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_058() {
        let n = (58 % 16) + 1;
        let t = Tensor::linspace(0.0, 58.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_059() {
        let n = (59 % 16) + 1;
        let t = Tensor::linspace(0.0, 59.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_060() {
        let n = (60 % 16) + 1;
        let t = Tensor::linspace(0.0, 60.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_061() {
        let n = (61 % 16) + 1;
        let t = Tensor::linspace(0.0, 61.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_062() {
        let n = (62 % 16) + 1;
        let t = Tensor::linspace(0.0, 62.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_063() {
        let n = (63 % 16) + 1;
        let t = Tensor::linspace(0.0, 63.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_064() {
        let n = (64 % 16) + 1;
        let t = Tensor::linspace(0.0, 64.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_065() {
        let n = (65 % 16) + 1;
        let t = Tensor::linspace(0.0, 65.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_066() {
        let n = (66 % 16) + 1;
        let t = Tensor::linspace(0.0, 66.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_067() {
        let n = (67 % 16) + 1;
        let t = Tensor::linspace(0.0, 67.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_068() {
        let n = (68 % 16) + 1;
        let t = Tensor::linspace(0.0, 68.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_069() {
        let n = (69 % 16) + 1;
        let t = Tensor::linspace(0.0, 69.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_070() {
        let n = (70 % 16) + 1;
        let t = Tensor::linspace(0.0, 70.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_071() {
        let n = (71 % 16) + 1;
        let t = Tensor::linspace(0.0, 71.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_072() {
        let n = (72 % 16) + 1;
        let t = Tensor::linspace(0.0, 72.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_073() {
        let n = (73 % 16) + 1;
        let t = Tensor::linspace(0.0, 73.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_074() {
        let n = (74 % 16) + 1;
        let t = Tensor::linspace(0.0, 74.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_075() {
        let n = (75 % 16) + 1;
        let t = Tensor::linspace(0.0, 75.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_076() {
        let n = (76 % 16) + 1;
        let t = Tensor::linspace(0.0, 76.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_077() {
        let n = (77 % 16) + 1;
        let t = Tensor::linspace(0.0, 77.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_078() {
        let n = (78 % 16) + 1;
        let t = Tensor::linspace(0.0, 78.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_079() {
        let n = (79 % 16) + 1;
        let t = Tensor::linspace(0.0, 79.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_080() {
        let n = (80 % 16) + 1;
        let t = Tensor::linspace(0.0, 80.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_081() {
        let n = (81 % 16) + 1;
        let t = Tensor::linspace(0.0, 81.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_082() {
        let n = (82 % 16) + 1;
        let t = Tensor::linspace(0.0, 82.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_083() {
        let n = (83 % 16) + 1;
        let t = Tensor::linspace(0.0, 83.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_084() {
        let n = (84 % 16) + 1;
        let t = Tensor::linspace(0.0, 84.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_085() {
        let n = (85 % 16) + 1;
        let t = Tensor::linspace(0.0, 85.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_086() {
        let n = (86 % 16) + 1;
        let t = Tensor::linspace(0.0, 86.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_087() {
        let n = (87 % 16) + 1;
        let t = Tensor::linspace(0.0, 87.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_088() {
        let n = (88 % 16) + 1;
        let t = Tensor::linspace(0.0, 88.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_089() {
        let n = (89 % 16) + 1;
        let t = Tensor::linspace(0.0, 89.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_090() {
        let n = (90 % 16) + 1;
        let t = Tensor::linspace(0.0, 90.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_091() {
        let n = (91 % 16) + 1;
        let t = Tensor::linspace(0.0, 91.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_092() {
        let n = (92 % 16) + 1;
        let t = Tensor::linspace(0.0, 92.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_093() {
        let n = (93 % 16) + 1;
        let t = Tensor::linspace(0.0, 93.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_094() {
        let n = (94 % 16) + 1;
        let t = Tensor::linspace(0.0, 94.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_095() {
        let n = (95 % 16) + 1;
        let t = Tensor::linspace(0.0, 95.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_096() {
        let n = (96 % 16) + 1;
        let t = Tensor::linspace(0.0, 96.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_097() {
        let n = (97 % 16) + 1;
        let t = Tensor::linspace(0.0, 97.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_098() {
        let n = (98 % 16) + 1;
        let t = Tensor::linspace(0.0, 98.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_099() {
        let n = (99 % 16) + 1;
        let t = Tensor::linspace(0.0, 99.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_100() {
        let n = (100 % 16) + 1;
        let t = Tensor::linspace(0.0, 100.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_101() {
        let n = (101 % 16) + 1;
        let t = Tensor::linspace(0.0, 101.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_102() {
        let n = (102 % 16) + 1;
        let t = Tensor::linspace(0.0, 102.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_103() {
        let n = (103 % 16) + 1;
        let t = Tensor::linspace(0.0, 103.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_104() {
        let n = (104 % 16) + 1;
        let t = Tensor::linspace(0.0, 104.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_105() {
        let n = (105 % 16) + 1;
        let t = Tensor::linspace(0.0, 105.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_106() {
        let n = (106 % 16) + 1;
        let t = Tensor::linspace(0.0, 106.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_107() {
        let n = (107 % 16) + 1;
        let t = Tensor::linspace(0.0, 107.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_108() {
        let n = (108 % 16) + 1;
        let t = Tensor::linspace(0.0, 108.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_109() {
        let n = (109 % 16) + 1;
        let t = Tensor::linspace(0.0, 109.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_110() {
        let n = (110 % 16) + 1;
        let t = Tensor::linspace(0.0, 110.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_111() {
        let n = (111 % 16) + 1;
        let t = Tensor::linspace(0.0, 111.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_112() {
        let n = (112 % 16) + 1;
        let t = Tensor::linspace(0.0, 112.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_113() {
        let n = (113 % 16) + 1;
        let t = Tensor::linspace(0.0, 113.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_114() {
        let n = (114 % 16) + 1;
        let t = Tensor::linspace(0.0, 114.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_115() {
        let n = (115 % 16) + 1;
        let t = Tensor::linspace(0.0, 115.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_116() {
        let n = (116 % 16) + 1;
        let t = Tensor::linspace(0.0, 116.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_117() {
        let n = (117 % 16) + 1;
        let t = Tensor::linspace(0.0, 117.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_118() {
        let n = (118 % 16) + 1;
        let t = Tensor::linspace(0.0, 118.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_119() {
        let n = (119 % 16) + 1;
        let t = Tensor::linspace(0.0, 119.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_120() {
        let n = (120 % 16) + 1;
        let t = Tensor::linspace(0.0, 120.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_121() {
        let n = (121 % 16) + 1;
        let t = Tensor::linspace(0.0, 121.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_122() {
        let n = (122 % 16) + 1;
        let t = Tensor::linspace(0.0, 122.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_123() {
        let n = (123 % 16) + 1;
        let t = Tensor::linspace(0.0, 123.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_124() {
        let n = (124 % 16) + 1;
        let t = Tensor::linspace(0.0, 124.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_125() {
        let n = (125 % 16) + 1;
        let t = Tensor::linspace(0.0, 125.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_126() {
        let n = (126 % 16) + 1;
        let t = Tensor::linspace(0.0, 126.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_127() {
        let n = (127 % 16) + 1;
        let t = Tensor::linspace(0.0, 127.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_128() {
        let n = (128 % 16) + 1;
        let t = Tensor::linspace(0.0, 128.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_129() {
        let n = (129 % 16) + 1;
        let t = Tensor::linspace(0.0, 129.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_130() {
        let n = (130 % 16) + 1;
        let t = Tensor::linspace(0.0, 130.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_131() {
        let n = (131 % 16) + 1;
        let t = Tensor::linspace(0.0, 131.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_132() {
        let n = (132 % 16) + 1;
        let t = Tensor::linspace(0.0, 132.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_133() {
        let n = (133 % 16) + 1;
        let t = Tensor::linspace(0.0, 133.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_134() {
        let n = (134 % 16) + 1;
        let t = Tensor::linspace(0.0, 134.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_135() {
        let n = (135 % 16) + 1;
        let t = Tensor::linspace(0.0, 135.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_136() {
        let n = (136 % 16) + 1;
        let t = Tensor::linspace(0.0, 136.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_137() {
        let n = (137 % 16) + 1;
        let t = Tensor::linspace(0.0, 137.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_138() {
        let n = (138 % 16) + 1;
        let t = Tensor::linspace(0.0, 138.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_139() {
        let n = (139 % 16) + 1;
        let t = Tensor::linspace(0.0, 139.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_140() {
        let n = (140 % 16) + 1;
        let t = Tensor::linspace(0.0, 140.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_141() {
        let n = (141 % 16) + 1;
        let t = Tensor::linspace(0.0, 141.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_142() {
        let n = (142 % 16) + 1;
        let t = Tensor::linspace(0.0, 142.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_143() {
        let n = (143 % 16) + 1;
        let t = Tensor::linspace(0.0, 143.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_144() {
        let n = (144 % 16) + 1;
        let t = Tensor::linspace(0.0, 144.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_145() {
        let n = (145 % 16) + 1;
        let t = Tensor::linspace(0.0, 145.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_146() {
        let n = (146 % 16) + 1;
        let t = Tensor::linspace(0.0, 146.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_147() {
        let n = (147 % 16) + 1;
        let t = Tensor::linspace(0.0, 147.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_148() {
        let n = (148 % 16) + 1;
        let t = Tensor::linspace(0.0, 148.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_149() {
        let n = (149 % 16) + 1;
        let t = Tensor::linspace(0.0, 149.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_150() {
        let n = (150 % 16) + 1;
        let t = Tensor::linspace(0.0, 150.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_151() {
        let n = (151 % 16) + 1;
        let t = Tensor::linspace(0.0, 151.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_152() {
        let n = (152 % 16) + 1;
        let t = Tensor::linspace(0.0, 152.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_153() {
        let n = (153 % 16) + 1;
        let t = Tensor::linspace(0.0, 153.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_154() {
        let n = (154 % 16) + 1;
        let t = Tensor::linspace(0.0, 154.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_155() {
        let n = (155 % 16) + 1;
        let t = Tensor::linspace(0.0, 155.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_156() {
        let n = (156 % 16) + 1;
        let t = Tensor::linspace(0.0, 156.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_157() {
        let n = (157 % 16) + 1;
        let t = Tensor::linspace(0.0, 157.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_158() {
        let n = (158 % 16) + 1;
        let t = Tensor::linspace(0.0, 158.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }

    #[test]
    fn test_tensor_stress_case_159() {
        let n = (159 % 16) + 1;
        let t = Tensor::linspace(0.0, 159.0, n * 2);
        assert_eq!(t.numel(), n * 2);
        let reshaped = t.reshape(vec![2, n]);
        assert_eq!(reshaped.shape(), &[2, n]);
        let tr = reshaped.t();
        assert_eq!(tr.shape(), &[n, 2]);
        assert_eq!(tr.get_2d(0, 0), 0.0);
        
        let sub = tr.narrow(0, 0, 1);
        assert_eq!(sub.shape(), &[1, 2]);
    }
}
