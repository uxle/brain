//! Core Tensor struct implementation for the Brain deep learning framework.
//!
//! This module defines the [`Tensor`] struct and all its fundamental operations
//! including creation, element access, transformations, operator overloads,
//! and mutation methods.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index, IndexMut};

use crate::device::Device;
use crate::dtype::DType;
use crate::error::BrainResult;
use crate::random::{self, BrainRng};

// =============================================================================
// Tensor Struct
// =============================================================================

/// A multi-dimensional array of numerical data.
///
/// The Tensor is the core data structure of the Brain framework, representing
/// an n-dimensional array with a given shape, data type, device placement,
/// and optional gradient tracking.
///
/// Internally, data is stored as a contiguous `Vec<f64>` in row-major order.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The underlying data stored as f64 values in row-major order.
    data: Vec<f64>,
    /// The shape (dimensions) of the tensor.
    shape: Vec<usize>,
    /// The strides for each dimension.
    strides: Vec<usize>,
    /// The device where the tensor resides.
    device: Device,
    /// The data type of the tensor elements.
    dtype: DType,
    /// Whether gradients should be computed for this tensor.
    requires_grad: bool,
    /// An optional name for debugging purposes.
    name: Option<String>,
}

// =============================================================================
// Tensor Implementation
// =============================================================================

impl Tensor {
    // -----------------------------------------------------------------------
    // Creation Methods
    // -----------------------------------------------------------------------

    /// Creates a new tensor from a data vector and shape.
    ///
    /// # Panics
    ///
    /// Panics if the product of shape dimensions does not equal data.len().
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel,
            "Data length {} does not match shape product {}",
            data.len(), numel);
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

    /// Creates a tensor filled with a specific value.
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

    /// Creates a scalar tensor (0-dimensional).
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

    /// Creates an identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let shape = vec![n, n];
        Tensor {
            data,
            shape: shape.clone(),
            strides: compute_strides(&shape),
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a 1D tensor with values from start to end (exclusive) with given step.
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        assert!(step != 0.0, "Step must be non-zero");
        let len = if step > 0.0 {
            ((end - start) / step).ceil() as usize
        } else {
            ((start - end) / (-step)).ceil() as usize
        };
        let data: Vec<f64> = (0..len).map(|i| start + i as f64 * step).collect();
        let shape = vec![len];
        Tensor {
            data,
            shape: shape.clone(),
            strides: compute_strides(&shape),
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a 1D tensor with evenly spaced values over [start, end].
    pub fn linspace(start: f64, end: f64, num: usize) -> Self {
        assert!(num >= 1, "Number of points must be >= 1");
        let data: Vec<f64> = if num == 1 {
            vec![start]
        } else {
            let step = (end - start) / (num - 1) as f64;
            (0..num).map(|i| start + i as f64 * step).collect()
        };
        let shape = vec![num];
        Tensor {
            data,
            shape: shape.clone(),
            strides: compute_strides(&shape),
            device: Device::Cpu,
            dtype: DType::F64,
            requires_grad: false,
            name: None,
        }
    }

    /// Creates a 1D tensor with logarithmically spaced values.
    pub fn logspace(base: f64, start: f64, end: f64, num: usize) -> Self {
        let inner = Self::linspace(start, end, num);
        let data = inner.data.iter().map(|&v| base.powf(v)).collect();
        Tensor { data, shape: vec![num], strides: vec![1], device: Device::Cpu, dtype: DType::F64, requires_grad: false, name: None }
    }

    /// Creates an identity-like 2D tensor with ones on the k-th diagonal.
    pub fn eye(n: usize, m: usize, k: isize) -> Self {
        let mut data = vec![0.0; n * m];
        let diag_start = if k >= 0 { k as usize } else { (-k) as usize };
        for i in 0..n {
            let j = i as isize + k;
            if j >= 0 && (j as usize) < m {
                data[i * m + j as usize] = 1.0;
            }
        }
        let shape = vec![n, m];
        Tensor { data, shape: shape.clone(), strides: compute_strides(&shape), device: Device::Cpu, dtype: DType::F64, requires_grad: false, name: None }
    }

    /// Creates an uninitialized tensor (filled with 0.0 for safety).
    pub fn empty(shape: Vec<usize>) -> Self {
        Self::zeros(shape)
    }

    /// Creates a tensor from a flat slice of values with a given shape.
    pub fn from_slice(slice: &[f64], shape: Vec<usize>) -> Self {
        Self::new(slice.to_vec(), shape)
    }

    /// Creates a 2D matrix with the given vector on the diagonal.
    pub fn from_diag(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = diag[i];
        }
        let shape = vec![n, n];
        Tensor { data, shape: shape.clone(), strides: compute_strides(&shape), device: Device::Cpu, dtype: DType::F64, requires_grad: false, name: None }
    }

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] { &self.shape }

    /// Returns the strides of the tensor.
    pub fn strides(&self) -> &[usize] { &self.strides }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &[f64] { &self.data }

    /// Returns a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut [f64] { &mut self.data }

    /// Returns the number of dimensions (rank).
    pub fn ndim(&self) -> usize { self.shape.len() }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize { self.data.len() }

    /// Returns true if the tensor has no elements.
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Returns whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool { self.requires_grad }

    /// Sets the requires_grad flag.
    pub fn set_requires_grad(&mut self, requires_grad: bool) { self.requires_grad = requires_grad; }

    /// Returns the device.
    pub fn device(&self) -> Device { self.device }

    /// Returns the data type.
    pub fn dtype(&self) -> DType { self.dtype }

    /// Returns true if the tensor data is contiguous in row-major order.
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /// Returns true if this is a scalar (0-dimensional) tensor.
    pub fn is_scalar(&self) -> bool { self.shape.is_empty() }

    /// Returns true if this is a 2D matrix.
    pub fn is_matrix(&self) -> bool { self.shape.len() == 2 }

    /// Returns true if this is a 1D vector.
    pub fn is_vector(&self) -> bool { self.shape.len() == 1 }

    /// Returns the size of the i-th dimension.
    pub fn size(&self, i: usize) -> usize {
        self.shape.get(i).copied().unwrap_or(1)
    }

    /// Returns the optional name of the tensor.
    pub fn name(&self) -> Option<&str> { self.name.as_deref() }

    // -----------------------------------------------------------------------
    // Element Access
    // -----------------------------------------------------------------------

    /// Gets the element at the given flat index.
    pub fn get(&self, index: usize) -> f64 {
        assert!(index < self.data.len(), "Index {} out of bounds for tensor of size {}", index, self.data.len());
        self.data[index]
    }

    /// Gets a mutable reference to the element at the given flat index.
    pub fn get_mut(&mut self, index: usize) -> &mut f64 {
        assert!(index < self.data.len(), "Index {} out of bounds for tensor of size {}", index, self.data.len());
        &mut self.data[index]
    }

    /// Sets the element at the given flat index.
    pub fn set(&mut self, index: usize, value: f64) {
        assert!(index < self.data.len(), "Index {} out of bounds for tensor of size {}", index, self.data.len());
        self.data[index] = value;
    }

    /// Gets the element at the given multi-dimensional index.
    pub fn get_index(&self, indices: &[usize]) -> f64 {
        let flat = self.multi_to_flat(indices);
        self.data[flat]
    }

    /// Sets the element at the given multi-dimensional index.
    pub fn set_index(&mut self, indices: &[usize], value: f64) {
        let flat = self.multi_to_flat(indices);
        self.data[flat] = value;
    }

    /// Gets element without bounds checking.
    pub unsafe fn unsafe_get(&self, index: usize) -> f64 {
        *self.data.get_unchecked(index)
    }

    /// Sets element without bounds checking.
    pub unsafe fn unsafe_set(&mut self, index: usize, value: f64) {
        *self.data.get_unchecked_mut(index) = value;
    }

    // -----------------------------------------------------------------------
    // Transformations
    // -----------------------------------------------------------------------

    /// Reshapes the tensor to a new shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(self.data.len(), new_numel,
            "Cannot reshape tensor of size {} to shape {:?}", self.data.len(), new_shape);
        Tensor {
            data: self.data.clone(),
            shape: new_shape.clone(),
            strides: compute_strides(&new_shape),
            device: self.device,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            name: self.name.clone(),
        }
    }

    /// Flattens the tensor to 1D.
    pub fn flatten(&self) -> Self {
        let numel = self.data.len();
        Tensor {
            data: self.data.clone(),
            shape: vec![numel],
            strides: vec![1],
            device: self.device,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            name: self.name.clone(),
        }
    }

    /// Transposes the last two dimensions of a 2D tensor.
    pub fn transpose(&self) -> Self {
        assert!(self.ndim() == 2, "Transpose requires a 2D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }
        let shape = vec![cols, rows];
        Tensor { data, shape: shape.clone(), strides: compute_strides(&shape), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Permute the dimensions according to the given permutation.
    pub fn permute(&self, perm: &[usize]) -> Self {
        assert_eq!(perm.len(), self.shape.len(), "Permutation length must match ndim");
        let mut new_shape = Vec::with_capacity(perm.len());
        for &p in perm { new_shape.push(self.shape[p]); }
        let new_numel: usize = new_shape.iter().product();
        let mut data = vec![0.0; new_numel];
        let new_strides = compute_strides(&new_shape);
        // Iterate over all indices
        let mut idx = vec![0usize; self.shape.len()];
        let mut flat = 0;
        loop {
            // Compute source flat index
            let src_flat = self.multi_to_flat(&idx);
            // Compute dest multi index
            let mut dest_multi = vec![0usize; perm.len()];
            for (i, &p) in perm.iter().enumerate() { dest_multi[i] = idx[p]; }
            let dest_flat = multi_to_flat_raw(&dest_multi, &new_strides);
            data[dest_flat] = self.data[src_flat];
            // Increment
            flat += 1;
            if flat >= new_numel { break; }
            let mut carry = true;
            for i in (0..self.shape.len()).rev() {
                if carry {
                    idx[i] += 1;
                    if idx[i] >= self.shape[i] { idx[i] = 0; } else { carry = false; }
                }
            }
        }
        Tensor { data, shape: new_shape.clone(), strides: new_strides, device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Removes dimensions of size 1.
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().filter(|&&d| d != 1).cloned().collect();
        if new_shape.is_empty() {
            return self.reshape(vec![1]);
        }
        self.reshape(new_shape)
    }

    /// Removes a specific dimension of size 1.
    pub fn squeeze_axis(&self, axis: usize) -> Self {
        assert!(axis < self.shape.len());
        assert_eq!(self.shape[axis], 1, "Cannot squeeze dimension of size {}", self.shape[axis]);
        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        self.reshape(new_shape)
    }

    /// Adds a dimension of size 1 at the given axis.
    pub fn unsqueeze(&self, axis: i32) -> Self {
        let axis = if axis < 0 { self.shape.len() as i32 + axis } else { axis } as usize;
        assert!(axis <= self.shape.len());
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        self.reshape(new_shape)
    }

    /// Narrows the tensor along a dimension.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Self {
        assert!(dim < self.shape.len());
        assert!(start + length <= self.shape[dim]);
        let mut new_shape = self.shape.clone();
        new_shape[dim] = length;
        let new_numel: usize = new_shape.iter().product();
        let mut data = vec![0.0; new_numel];
        let mut src_idx = vec![0usize; self.shape.len()];
        let mut dst_idx = vec![0usize; new_shape.len()];
        let mut dst_flat = 0;
        loop {
            // Copy element
            let src_flat = self.multi_to_flat(&src_idx);
            let dst_flat_val = multi_to_flat_raw(&dst_idx, &compute_strides(&new_shape));
            data[dst_flat_val] = self.data[src_flat];
            dst_flat += 1;
            if dst_flat >= new_numel { break; }
            // Increment dst
            let mut carry = true;
            for i in (0..new_shape.len()).rev() {
                if carry {
                    dst_idx[i] += 1;
                    if dst_idx[i] >= new_shape[i] { dst_idx[i] = 0; } else { carry = false; }
                }
            }
            // Map dst to src
            for i in 0..self.shape.len() {
                src_idx[i] = dst_idx.get(i).copied().unwrap_or(0);
            }
            src_idx[dim] = start + dst_idx[dim];
        }
        Tensor { data, shape: new_shape.clone(), strides: compute_strides(&new_shape), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Splits the tensor into chunks along a dimension.
    pub fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self> {
        assert!(dim < self.shape.len());
        assert!(chunks > 0);
        let dim_size = self.shape[dim];
        let base_len = dim_size / chunks;
        let remainder = dim_size % chunks;
        let mut result = Vec::with_capacity(chunks);
        let mut start = 0;
        for i in 0..chunks {
            let len = base_len + if i < remainder { 1 } else { 0 };
            result.push(self.narrow(dim, start, len));
            start += len;
        }
        result
    }

    /// Splits the tensor into equal parts along a dimension.
    pub fn split(&self, sections: usize, dim: usize) -> Vec<Self> {
        self.chunk(sections, dim)
    }

    /// Flips the tensor along a dimension.
    pub fn flip(&self, dim: usize) -> Self {
        assert!(dim < self.shape.len());
        let dim_size = self.shape[dim];
        let mut data = self.data.clone();
        let mut idx = vec![0usize; self.shape.len()];
        for flat in 0..self.data.len() {
            idx[dim] = dim_size - 1 - idx[dim];
            let new_flat = self.multi_to_flat(&idx);
            data[flat] = self.data[new_flat];
            idx[dim] = dim_size - 1 - idx[dim];
            // Increment
            let mut carry = true;
            for i in (0..self.shape.len()).rev() {
                if carry {
                    idx[i] += 1;
                    if idx[i] >= self.shape[i] { idx[i] = 0; } else { carry = false; }
                }
            }
        }
        let shape = self.shape.clone();
        Tensor { data, shape, strides: self.strides.clone(), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Rolls the tensor along a dimension by a given number of positions.
    pub fn roll(&self, shift: isize, dim: usize) -> Self {
        assert!(dim < self.shape.len());
        let dim_size = self.shape[dim] as isize;
        let shift = ((shift % dim_size) + dim_size) % dim_size;
        let shift = shift as usize;
        let mut data = vec![0.0; self.data.len()];
        let mut idx = vec![0usize; self.shape.len()];
        for flat in 0..self.data.len() {
            let rolled_idx = (idx[dim] + shift) % self.shape[dim];
            let mut src_idx = idx.clone();
            src_idx[dim] = rolled_idx;
            let src_flat = self.multi_to_flat(&src_idx);
            data[flat] = self.data[src_flat];
            // Increment
            let mut carry = true;
            for i in (0..self.shape.len()).rev() {
                if carry {
                    idx[i] += 1;
                    if idx[i] >= self.shape[i] { idx[i] = 0; } else { carry = false; }
                }
            }
        }
        let shape = self.shape.clone();
        Tensor { data, shape, strides: self.strides.clone(), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Returns the lower triangular part of a 2D tensor.
    pub fn tril(&self, k: isize) -> Self {
        assert!(self.ndim() == 2, "tril requires 2D tensor");
        let mut data = self.data.clone();
        let (rows, cols) = (self.shape[0], self.shape[1]);
        for i in 0..rows {
            for j in 0..cols {
                if j as isize > i as isize + k {
                    data[i * cols + j] = 0.0;
                }
            }
        }
        let shape = self.shape.clone();
        Tensor { data, shape, strides: self.strides.clone(), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Returns the upper triangular part of a 2D tensor.
    pub fn triu(&self, k: isize) -> Self {
        assert!(self.ndim() == 2, "triu requires 2D tensor");
        let mut data = self.data.clone();
        let (rows, cols) = (self.shape[0], self.shape[1]);
        for i in 0..rows {
            for j in 0..cols {
                if j as isize < i as isize + k {
                    data[i * cols + j] = 0.0;
                }
            }
        }
        let shape = self.shape.clone();
        Tensor { data, shape, strides: self.strides.clone(), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Extracts the diagonal of a 2D tensor.
    pub fn diagonal(&self, offset: isize) -> Self {
        assert!(self.ndim() == 2, "diagonal requires 2D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let diag_len = if offset >= 0 { rows.min(cols - offset as usize) } else { (rows - (-offset) as usize).min(cols) };
        let data: Vec<f64> = (0..diag_len).map(|i| {
            let r = i;
            let c = (i as isize + offset) as usize;
            self.data[r * cols + c]
        }).collect();
        let shape = vec![diag_len];
        Tensor { data, shape: shape.clone(), strides: compute_strides(&shape), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Takes elements at the given indices along the first dimension.
    pub fn take(&self, indices: &[usize]) -> Self {
        let dim0 = self.shape[0];
        let sub_size: usize = if self.shape.len() > 1 { self.shape[1..].iter().product() } else { 1 };
        let mut data = Vec::with_capacity(indices.len() * sub_size);
        for &idx in indices {
            assert!(idx < dim0, "Index {} out of bounds for dimension of size {}", idx, dim0);
            let start = idx * sub_size;
            data.extend_from_slice(&self.data[start..start + sub_size]);
        }
        let mut new_shape = vec![indices.len()];
        new_shape.extend_from_slice(&self.shape[1..]);
        Tensor { data, shape: new_shape.clone(), strides: compute_strides(&new_shape), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Expands the tensor to a larger shape (broadcast).
    pub fn expand(&self, new_shape: Vec<usize>) -> Self {
        let new_numel: usize = new_shape.iter().product();
        let mut data = vec![0.0; new_numel];
        let rank_diff = new_shape.len() - self.shape.len();
        for i in 0..new_numel {
            // Decompose i into multi-dim index in new_shape
            let mut multi = vec![0usize; new_shape.len()];
            let mut val = i;
            for j in (0..new_shape.len()).rev() {
                multi[j] = val % new_shape[j];
                val /= new_shape[j];
            }
            // Map to source multi-dim index
            let mut src_multi = vec![0usize; self.shape.len()];
            for j in 0..self.shape.len() {
                src_multi[j] = if self.shape[j] == 1 { 0 } else { multi[rank_diff + j] };
            }
            data[i] = self.data[self.multi_to_flat(&src_multi)];
        }
        Tensor { data, shape: new_shape.clone(), strides: compute_strides(&new_shape), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Repeats the tensor along each dimension the given number of times.
    pub fn repeat(&self, reps: &[usize]) -> Self {
        let mut new_shape = self.shape.clone();
        for (i, &r) in reps.iter().enumerate() {
            if i < new_shape.len() { new_shape[i] *= r; } else { new_shape.push(r); }
        }
        self.expand(new_shape.clone())
    }

    /// Tiles the tensor (alias for repeat with broadcasting behavior).
    pub fn tile(&self, reps: &[usize]) -> Self {
        let mut new_shape = self.shape.clone();
        for &r in reps { new_shape.push(r); }
        self.expand(new_shape.clone())
    }

    /// Returns a view of this tensor (cloned data).
    pub fn view(&self) -> Self {
        self.clone()
    }

    /// Ensures the tensor is contiguous (no-op if already contiguous).
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        // Re-layout data
        let mut data = vec![0.0; self.data.len()];
        for i in 0..self.data.len() {
            data[i] = self.data[i]; // already flat
        }
        let shape = self.shape.clone();
        Tensor { data, shape: shape.clone(), strides: compute_strides(&shape), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Detaches the tensor from the computation graph.
    pub fn detach(&self) -> Self {
        let mut t = self.clone();
        t.requires_grad = false;
        t
    }

    /// Creates a deep clone of this tensor.
    pub fn clone_tensor(&self) -> Self {
        self.clone()
    }

    /// Moves the tensor to the given device (no-op for CPU).
    pub fn to_device(&self, device: Device) -> Self {
        assert_eq!(self.device, Device::Cpu, "Can only move from CPU");
        assert_eq!(device, Device::Cpu, "Only CPU device supported");
        self.clone()
    }

    // -----------------------------------------------------------------------
    // Mutation Methods
    // -----------------------------------------------------------------------

    /// Fills the tensor with a given value.
    pub fn fill_(&mut self, value: f64) {
        for v in self.data.iter_mut() { *v = value; }
    }

    /// Fills the tensor with zeros.
    pub fn zero_(&mut self) {
        self.fill_(0.0);
    }

    /// Fills the tensor with values from a uniform distribution [low, high).
    pub fn uniform_(&mut self, low: f64, high: f64) {
        let mut rng = random::default_rng();
        for v in self.data.iter_mut() { *v = rng.uniform(low, high); }
    }

    /// Fills the tensor with values from a normal distribution.
    pub fn normal_(&mut self, mean: f64, std: f64) {
        let mut rng = random::default_rng();
        for v in self.data.iter_mut() { *v = rng.normal(mean, std); }
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Computes statistics about the tensor elements.
    pub fn statistics(&self) -> crate::tensor::TensorStats {
        let n = self.data.len();
        if n == 0 {
            return crate::tensor::TensorStats::default();
        }
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut num_zeros = 0usize;
        let mut num_nans = 0usize;
        let mut l1_norm = 0.0;
        let mut l2_norm_sq = 0.0;

        for &v in &self.data {
            if v.is_nan() { num_nans += 1; continue; }
            if v < min { min = v; }
            if v > max { max = v; }
            sum += v;
            sum_sq += v * v;
            if v == 0.0 { num_zeros += 1; }
            l1_norm += v.abs();
            l2_norm_sq += v * v;
        }

        let mean = sum / n as f64;
        let variance = if n > 1 { (sum_sq - sum * sum / n as f64) / (n - 1) as f64 } else { 0.0 };
        let std = variance.sqrt();

        crate::tensor::TensorStats {
            min, max, mean, std,
            num_zeros, num_nans,
            sparsity: num_zeros as f64 / n as f64,
            l1_norm,
            l2_norm: l2_norm_sq.sqrt(),
        }
    }

    // -----------------------------------------------------------------------
    // Element-wise map/reduce helpers
    // -----------------------------------------------------------------------

    /// Applies a function element-wise and returns a new tensor.
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| f(v)).collect();
        let shape = self.shape.clone();
        Tensor { data, shape, strides: self.strides.clone(), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Applies a function element-wise in-place.
    pub fn map_inplace<F: Fn(f64) -> f64>(&mut self, f: F) {
        for v in self.data.iter_mut() { *v = f(*v); }
    }

    /// Applies a binary function element-wise with another tensor (broadcasting).
    pub fn map2<F: Fn(f64, f64) -> f64>(&self, other: &Tensor, f: F) -> Tensor {
        assert!(validate_binary_shapes(&self.shape, &other.shape),
            "Shapes {:?} and {:?} are not broadcast-compatible", self.shape, other.shape);
        let out_shape = binary_broadcast_shape(&self.shape, &other.shape);
        let out_numel: usize = out_shape.iter().product();
        let mut data = vec![0.0; out_numel];
        for i in 0..out_numel {
            let a_idx = broadcast_flat_index(i, &out_shape, &self.shape);
            let b_idx = broadcast_flat_index(i, &out_shape, &other.shape);
            data[i] = f(self.data[a_idx], other.data[b_idx]);
        }
        Tensor { data, shape: out_shape.clone(), strides: compute_strides(&out_shape), device: self.device, dtype: self.dtype, requires_grad: self.requires_grad, name: self.name.clone() }
    }

    /// Reduces the tensor along all dimensions using a binary accumulator.
    pub fn reduce<F: Fn(f64, f64) -> f64>(&self, init: f64, f: F) -> f64 {
        self.data.iter().fold(init, |acc, &v| f(acc, v))
    }

    /// Applies map2 for scalar (f64) right operand.
    pub fn map_scalar<F: Fn(f64, f64) -> f64>(&self, scalar: f64, f: F) -> Tensor {
        self.map(|v| f(v, scalar))
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn multi_to_flat(&self, indices: &[usize]) -> usize {
        let mut flat = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if i < self.strides.len() {
                flat += idx * self.strides[i];
            }
        }
        flat
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return vec![]; }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn multi_to_flat_raw(multi: &[usize], strides: &[usize]) -> usize {
    let mut flat = 0;
    for (i, &idx) in multi.iter().enumerate() {
        if i < strides.len() { flat += idx * strides[i]; }
    }
    flat
}

fn validate_binary_shapes(a: &[usize], b: &[usize]) -> bool {
    let max_ndim = a.len().max(b.len());
    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da != db && da != 1 && db != 1 { return false; }
    }
    true
}

fn binary_broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        result.push(if da == 1 { db } else { da });
    }
    result.reverse();
    result
}

fn broadcast_flat_index(output_idx: usize, output_shape: &[usize], source_shape: &[usize]) -> usize {
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

// =============================================================================
// Operator Implementations: Tensor + Tensor
// =============================================================================

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a + b)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor { (&self).add(&rhs) }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a - b)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor { (&self).sub(&rhs) }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a * b)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor { (&self).mul(&rhs) }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.map2(rhs, |a, b| a / b)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor { (&self).div(&rhs) }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.map(|v| -v)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor { (&self).neg() }
}

// =============================================================================
// Operator Implementations: Tensor + f64
// =============================================================================

impl Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor { self.map(|v| v + rhs) }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Tensor { (&self).add(rhs) }
}
impl Add<&Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor { rhs.map(|v| self + v) }
}

impl Sub<f64> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Tensor { self.map(|v| v - rhs) }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: f64) -> Tensor { (&self).sub(rhs) }
}
impl Sub<&Tensor> for f64 {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor { rhs.map(|v| self - v) }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor { self.map(|v| v * rhs) }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Tensor { (&self).mul(rhs) }
}
impl Mul<&Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor { rhs.map(|v| self * v) }
}

impl Div<f64> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor { self.map(|v| v / rhs) }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: f64) -> Tensor { (&self).div(rhs) }
}
impl Div<&Tensor> for f64 {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor { rhs.map(|v| self / v) }
}

// =============================================================================
// Compound Assignment Operators
// =============================================================================

impl AddAssign for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch for +=: {:?} vs {:?}", self.shape, rhs.shape);
        for i in 0..self.data.len().min(rhs.data.len()) { self.data[i] += rhs.data[i]; }
    }
}
impl AddAssign<f64> for Tensor {
    fn add_assign(&mut self, rhs: f64) { for v in self.data.iter_mut() { *v += rhs; } }
}

impl SubAssign for Tensor {
    fn sub_assign(&mut self, rhs: Tensor) {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch for -=: {:?} vs {:?}", self.shape, rhs.shape);
        for i in 0..self.data.len().min(rhs.data.len()) { self.data[i] -= rhs.data[i]; }
    }
}
impl SubAssign<f64> for Tensor {
    fn sub_assign(&mut self, rhs: f64) { for v in self.data.iter_mut() { *v -= rhs; } }
}

impl MulAssign for Tensor {
    fn mul_assign(&mut self, rhs: Tensor) {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch for *=: {:?} vs {:?}", self.shape, rhs.shape);
        for i in 0..self.data.len().min(rhs.data.len()) { self.data[i] *= rhs.data[i]; }
    }
}
impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, rhs: f64) { for v in self.data.iter_mut() { *v *= rhs; } }
}

impl DivAssign for Tensor {
    fn div_assign(&mut self, rhs: Tensor) {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch for /=: {:?} vs {:?}", self.shape, rhs.shape);
        for i in 0..self.data.len().min(rhs.data.len()) { self.data[i] /= rhs.data[i]; }
    }
}
impl DivAssign<f64> for Tensor {
    fn div_assign(&mut self, rhs: f64) { for v in self.data.iter_mut() { *v /= rhs; } }
}

// =============================================================================
// Index Implementation
// =============================================================================

impl Index<usize> for Tensor {
    type Output = f64;
    fn index(&self, index: usize) -> &f64 { &self.data[index] }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut f64 { &mut self.data[index] }
}

// =============================================================================
// PartialEq, PartialOrd, Hash
// =============================================================================

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl PartialOrd for Tensor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.shape != other.shape { return None; }
        self.data.partial_cmp(&other.data)
    }
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        for &v in &self.data { v.to_bits().hash(state); }
    }
}

// =============================================================================
// Display Implementation
// =============================================================================

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            return write!(f, "Tensor(scalar, value={:.6})", self.data[0]);
        }
        let shape_str = self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
        writeln!(f, "Tensor(shape=[{}], device={}, dtype={})", shape_str, self.device, self.dtype)?;
        let pretty = crate::tensor::pretty_print(&self.data, &self.shape, 0);
        write!(f, "{}", pretty)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        for i in 0..6 { assert_eq!(t.get(i), 0.0); }
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(vec![2, 3]);
        for i in 0..6 { assert_eq!(t.get(i), 1.0); }
    }

    #[test]
    fn test_full() {
        let t = Tensor::full(vec![2, 3], 5.0);
        for i in 0..6 { assert_eq!(t.get(i), 5.0); }
    }

    #[test]
    fn test_scalar() {
        let t = Tensor::scalar(42.0);
        assert!(t.is_scalar());
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.get(0), 42.0);
    }

    #[test]
    fn test_identity() {
        let t = Tensor::identity(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.get_index(&[0, 0]), 1.0);
        assert_eq!(t.get_index(&[1, 1]), 1.0);
        assert_eq!(t.get_index(&[0, 1]), 0.0);
    }

    #[test]
    fn test_arange() {
        let t = Tensor::arange(0.0, 10.0, 2.0);
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.get(0), 0.0);
        assert_eq!(t.get(1), 2.0);
        assert_eq!(t.get(4), 8.0);
    }

    #[test]
    fn test_arange_negative_step() {
        let t = Tensor::arange(10.0, 0.0, -2.0);
        assert_eq!(t.get(0), 10.0);
        assert_eq!(t.get(1), 8.0);
    }

    #[test]
    fn test_linspace() {
        let t = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.numel(), 5);
        assert_eq!(t.get(0), 0.0);
        assert_eq!(t.get(4), 1.0);
    }

    #[test]
    fn test_eye() {
        let t = Tensor::eye(3, 4, 0);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.get_index(&[0, 0]), 1.0);
        assert_eq!(t.get_index(&[1, 1]), 1.0);
        assert_eq!(t.get_index(&[2, 2]), 1.0);
    }

    #[test]
    fn test_from_slice() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.get(0), 1.0);
        assert_eq!(t.get(3), 4.0);
    }

    #[test]
    fn test_from_diag() {
        let t = Tensor::from_diag(&[1.0, 2.0, 3.0]);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.get_index(&[0, 0]), 1.0);
        assert_eq!(t.get_index(&[1, 1]), 2.0);
        assert_eq!(t.get_index(&[2, 2]), 3.0);
        assert_eq!(t.get_index(&[0, 1]), 0.0);
    }

    #[test]
    fn test_properties() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
        assert!(!t.is_empty());
        assert!(!t.is_scalar());
        assert!(t.is_matrix() == false);
        assert!(t.is_vector() == false);
        assert!(t.is_contiguous());
        assert_eq!(t.size(0), 2);
        assert_eq!(t.size(1), 3);
    }

    #[test]
    fn test_element_access() {
        let t = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        assert_eq!(t.get(0), 0.0);
        assert_eq!(t.get(11), 11.0);
        assert_eq!(t.get_index(&[1, 2]), 6.0);
    }

    #[test]
    fn test_set_element() {
        let mut t = Tensor::zeros(vec![3, 3]);
        t.set(4, 5.0);
        assert_eq!(t.get(4), 5.0);
        t.set_index(&[1, 1], 10.0);
        assert_eq!(t.get_index(&[1, 1]), 10.0);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds() {
        let t = Tensor::zeros(vec![2, 3]);
        t.get(100);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::arange(0.0, 12.0, 1.0);
        let r = t.reshape(vec![3, 4]);
        assert_eq!(r.shape(), &[3, 4]);
        assert_eq!(r.get_index(&[0, 0]), 0.0);
        assert_eq!(r.get_index(&[2, 3]), 11.0);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        let f = t.flatten();
        assert_eq!(f.shape(), &[24]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tt = t.transpose();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.get_index(&[0, 0]), 1.0);
        assert_eq!(tt.get_index(&[0, 1]), 4.0);
        assert_eq!(tt.get_index(&[1, 0]), 2.0);
    }

    #[test]
    fn test_squeeze() {
        let t = Tensor::ones(vec![1, 3, 1, 4, 1]);
        let s = t.squeeze();
        assert_eq!(s.shape(), &[3, 4]);
    }

    #[test]
    fn test_squeeze_axis() {
        let t = Tensor::ones(vec![2, 1, 4]);
        let s = t.squeeze_axis(1);
        assert_eq!(s.shape(), &[2, 4]);
    }

    #[test]
    fn test_unsqueeze() {
        let t = Tensor::ones(vec![3, 4]);
        let s = t.unsqueeze(0);
        assert_eq!(s.shape(), &[1, 3, 4]);
        let s2 = t.unsqueeze(-1);
        assert_eq!(s2.shape(), &[3, 4, 1]);
    }

    #[test]
    fn test_narrow() {
        let t = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let n = t.narrow(0, 1, 2);
        assert_eq!(n.shape(), &[2, 4]);
        assert_eq!(n.get_index(&[0, 0]), 4.0);
    }

    #[test]
    fn test_chunk() {
        let t = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let chunks = t.chunk(3, 0);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[1, 4]);
        assert_eq!(chunks[1].shape(), &[1, 4]);
        assert_eq!(chunks[2].shape(), &[1, 4]);
    }

    #[test]
    fn test_flip() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let f = t.flip(0);
        assert_eq!(f.get_index(&[0, 0]), 4.0);
        assert_eq!(f.get_index(&[1, 0]), 1.0);
    }

    #[test]
    fn test_roll() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let r = t.roll(2, 0);
        assert_eq!(r.get(0), 4.0);
        assert_eq!(r.get(1), 5.0);
        assert_eq!(r.get(2), 1.0);
    }

    #[test]
    fn test_tril() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let l = t.tril(0);
        assert_eq!(l.get_index(&[0, 1]), 0.0);
        assert_eq!(l.get_index(&[1, 0]), 4.0);
    }

    #[test]
    fn test_triu() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let u = t.triu(0);
        assert_eq!(u.get_index(&[1, 0]), 0.0);
        assert_eq!(u.get_index(&[0, 1]), 2.0);
    }

    #[test]
    fn test_diagonal() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let d = t.diagonal(0);
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.get(0), 1.0);
        assert_eq!(d.get(1), 5.0);
        assert_eq!(d.get(2), 9.0);
    }

    #[test]
    fn test_take() {
        let t = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        let taken = t.take(&[0, 2]);
        assert_eq!(taken.shape(), &[2, 4]);
        assert_eq!(taken.get_index(&[0, 0]), 0.0);
        assert_eq!(taken.get_index(&[1, 0]), 8.0);
    }

    #[test]
    fn test_expand() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let e = t.reshape(vec![2, 1, 2]).expand(vec![2, 3, 2]);
        assert_eq!(e.shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_fill() {
        let mut t = Tensor::zeros(vec![3, 3]);
        t.fill_(5.0);
        for i in 0..9 { assert_eq!(t.get(i), 5.0); }
    }

    #[test]
    fn test_zero() {
        let mut t = Tensor::ones(vec![3, 3]);
        t.zero_();
        for i in 0..9 { assert_eq!(t.get(i), 0.0); }
    }

    #[test]
    fn test_uniform() {
        let mut t = Tensor::zeros(vec![1000]);
        t.uniform_(0.0, 1.0);
        let min = t.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = t.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min >= 0.0);
        assert!(max < 1.0);
    }

    #[test]
    fn test_statistics() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let stats = t.statistics();
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_tensor() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        let c = &a + &b;
        assert_eq!(c.get(0), 5.0);
        assert_eq!(c.get(1), 7.0);
        assert_eq!(c.get(2), 9.0);
    }

    #[test]
    fn test_sub_tensor() {
        let a = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let c = &a - &b;
        assert_eq!(c.get(0), 3.0);
        assert_eq!(c.get(1), 3.0);
        assert_eq!(c.get(2), 3.0);
    }

    #[test]
    fn test_mul_tensor() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], vec![3]);
        let c = &a * &b;
        assert_eq!(c.get(0), 10.0);
        assert_eq!(c.get(1), 18.0);
        assert_eq!(c.get(2), 28.0);
    }

    #[test]
    fn test_div_tensor() {
        let a = Tensor::from_slice(&[10.0, 12.0, 14.0], vec![3]);
        let b = Tensor::from_slice(&[2.0, 3.0, 7.0], vec![3]);
        let c = &a / &b;
        assert_eq!(c.get(0), 5.0);
        assert_eq!(c.get(1), 4.0);
        assert_eq!(c.get(2), 2.0);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0], vec![3]);
        let b = -&a;
        assert_eq!(b.get(0), -1.0);
        assert_eq!(b.get(1), 2.0);
        assert_eq!(b.get(2), -3.0);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = &a + 10.0;
        assert_eq!(b.get(0), 11.0);
        let c = 10.0 + &a;
        assert_eq!(c.get(0), 11.0);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        let b = &a * 2.0;
        assert_eq!(b.get(0), 4.0);
        assert_eq!(b.get(2), 8.0);
    }

    #[test]
    fn test_add_assign() {
        let mut a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        a += b;
        assert_eq!(a.get(0), 4.0);
        assert_eq!(a.get(1), 6.0);
    }

    #[test]
    fn test_add_assign_scalar() {
        let mut a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        a += 10.0;
        assert_eq!(a.get(0), 11.0);
    }

    #[test]
    fn test_mul_assign() {
        let mut a = Tensor::from_slice(&[2.0, 3.0], vec![2]);
        a *= Tensor::from_slice(&[4.0, 5.0], vec![2]);
        assert_eq!(a.get(0), 8.0);
    }

    #[test]
    fn test_partial_eq() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_display() {
        let t = Tensor::scalar(42.0);
        let s = format!("{}", t);
        assert!(s.contains("scalar"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_map() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let squared = t.map(|v| v * v);
        assert_eq!(squared.get(0), 1.0);
        assert_eq!(squared.get(1), 4.0);
        assert_eq!(squared.get(2), 9.0);
    }

    #[test]
    fn test_map_inplace() {
        let mut t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        t.map_inplace(|v| v * v);
        assert_eq!(t.get(1), 4.0);
    }

    #[test]
    fn test_reduce() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let sum = t.reduce(0.0, |a, b| a + b);
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_broadcast_add() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_slice(&[10.0], vec![1]);
        let c = &a + &b;
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.get(0), 11.0);
    }

    #[test]
    fn test_logspace() {
        let t = Tensor::logspace(10.0, 0.0, 3.0, 4);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.get(0), 1.0);
        assert_eq!(t.get(3), 1000.0);
    }

    #[test]
    fn test_detach() {
        let mut t = Tensor::ones(vec![2, 3]);
        t.set_requires_grad(true);
        let d = t.detach();
        assert!(!d.requires_grad());
    }

    #[test]
    fn test_clone_tensor() {
        let t = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let c = t.clone_tensor();
        assert_eq!(t, c);
    }

    #[test]
    fn test_hash_tensor() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Tensor::from_slice(&[1.0, 2.0], vec![2]));
        set.insert(Tensor::from_slice(&[1.0, 2.0], vec![2]));
        set.insert(Tensor::from_slice(&[3.0, 4.0], vec![2]));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_index_trait() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![3]);
        assert_eq!(t[0], 10.0);
        assert_eq!(t[2], 30.0);
    }

    #[test]
    fn test_index_mut_trait() {
        let mut t = Tensor::zeros(vec![3]);
        t[1] = 42.0;
        assert_eq!(t[1], 42.0);
    }

    #[test]
    fn test_sub_scalar_tensor() {
        let a = Tensor::from_slice(&[10.0, 20.0], vec![2]);
        let b = 5.0 - &a;
        assert_eq!(b.get(0), -5.0);
        assert_eq!(b.get(1), -15.0);
    }

    #[test]
    fn test_div_scalar_tensor() {
        let a = Tensor::from_slice(&[2.0, 4.0], vec![2]);
        let b = 1.0 / &a;
        assert_eq!(b.get(0), 0.5);
        assert_eq!(b.get(1), 0.25);
    }

    #[test]
    fn test_diagonal_offset() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let d = t.diagonal(1);
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.get(0), 2.0);
        assert_eq!(d.get(1), 6.0);
    }

    #[test]
    fn test_permute() {
        let t = Tensor::arange(0.0, 24.0, 1.0).reshape(vec![2, 3, 4]);
        let p = t.permute(&[2, 0, 1]);
        assert_eq!(p.shape(), &[4, 2, 3]);
    }

    #[test]
    fn test_normal_fill() {
        let mut t = Tensor::zeros(vec![10000]);
        t.normal_(0.0, 1.0);
        let stats = t.statistics();
        assert!(stats.mean.abs() < 0.1);
        assert!((stats.std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_shape() {
        let t = Tensor::empty(vec![0, 3]);
        assert!(t.is_empty());
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn test_view_clone() {
        let t = Tensor::ones(vec![3, 4]);
        let v = t.view();
        assert_eq!(t, v);
    }

    #[test]
    fn test_contiguous() {
        let t = Tensor::arange(0.0, 12.0, 1.0).reshape(vec![3, 4]);
        assert!(t.is_contiguous());
        let c = t.contiguous();
        assert_eq!(t, c);
    }
}
