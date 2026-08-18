//! Shape manipulation and multi-dimensional indexing for the Brain deep learning framework.
//!
//! This module defines the [`Shape`] struct for managing tensor dimensions and provides
//! utilities for broadcasting, reshaping, stride computation, and advanced indexing.
//!
//! # Shape Operations
//!
//! * Creation from dimensions, slices, or flat data
//! * Broadcasting two shapes to a common output shape
//! * Computing strides for row-major (C) and column-major (Fortran) layouts
//! * Squeezing, expanding, and inserting dimensions
//! * Multi-dimensional indexing with the [`ShapeIndex`] struct
//! * Convolution, pooling, and transpose output shape computation
//!
//! # Usage
//!
//! ```
//! use brain_core::shape::{broadcast_shapes, Shape};
//!
//! let shape = Shape::from_dims(&[2, 3, 4]);
//! assert_eq!(shape.ndim(), 3);
//! assert_eq!(shape.numel(), 24);
//!
//! let broadcast = broadcast_shapes(&[2, 1], &[1, 3]);
//! assert_eq!(broadcast, vec![2, 3]);
//! ```

use std::fmt;

/// Dimension size type alias.
pub type Dim = usize;
/// Strides vector type alias.
pub type Strides = Vec<usize>;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use crate::error::{BrainError, BrainResult};

// =============================================================================
// Shape Struct
// =============================================================================

/// Represents the shape (dimensions) of a tensor.
///
/// A `Shape` wraps a `Vec<usize>` where each element is the size of the
/// corresponding dimension. The number of elements in the vector is the
/// rank (number of dimensions) of the tensor.
///
/// # Invariants
///
/// * All dimension sizes must be non-zero (no dimension of size 0, except
///   for tensors with zero total elements via a 0-sized dimension)
/// * The shape must be compatible with the total number of elements
///
/// # Memory Layout
///
/// `Shape` does not store stride information. Use [`row_major_strides`]
/// and [`col_major_strides`] to compute strides from a shape.
///
/// # Examples
///
/// ```
/// use brain_core::shape::Shape;
///
/// let s = Shape::from_dims(&[2, 3, 4]);
/// assert_eq!(s.ndim(), 3);
/// assert_eq!(s.numel(), 24);
/// assert_eq!(s.as_slice(), &[2, 3, 4]);
///
/// let squeezed = s.squeeze_shape(&[1]);
/// assert_eq!(squeezed.as_slice(), &[2, 3, 4]); // dim 1 is already > 1
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

// =============================================================================
// Shape Constructors
// =============================================================================

impl Shape {
    /// Creates a new Shape from a vector of dimension sizes.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension sizes
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(s.ndim(), 3);
    /// ```
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    /// Creates a Shape from a slice of dimension sizes.
    ///
    /// # Arguments
    ///
    /// * `dims` - Slice of dimension sizes
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3]);
    /// assert_eq!(s.numel(), 6);
    /// ```
    pub fn from_dims(dims: &[usize]) -> Self {
        Shape(dims.to_vec())
    }

    /// Creates a scalar shape (0-dimensional tensor).
    ///
    /// A scalar has no dimensions and exactly one element.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::scalar();
    /// assert_eq!(s.ndim(), 0);
    /// assert_eq!(s.numel(), 1);
    /// ```
    pub fn scalar() -> Self {
        Shape(Vec::new())
    }

    /// Creates a 1-dimensional shape (vector).
    ///
    /// # Arguments
    ///
    /// * `size` - The length of the vector
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::vector(10);
    /// assert_eq!(s.ndim(), 1);
    /// assert_eq!(s.size_at(0), 10);
    /// ```
    pub fn vector(size: usize) -> Self {
        Shape(vec![size])
    }

    /// Creates a 2-dimensional shape (matrix).
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::matrix(3, 4);
    /// assert_eq!(s.ndim(), 2);
    /// assert_eq!(s.numel(), 12);
    /// ```
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Shape(vec![rows, cols])
    }

    /// Creates a shape with a given rank where all dimensions are the same size.
    ///
    /// # Arguments
    ///
    /// * `rank` - Number of dimensions
    /// * `size` - Size for each dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::cube(3, 4);
    /// assert_eq!(s.as_slice(), &[4, 4, 4]);
    /// ```
    pub fn cube(rank: usize, size: usize) -> Self {
        Shape(vec![size; rank])
    }

    /// Creates a batch shape with the given batch dimensions followed by
    /// feature dimensions.
    ///
    /// # Arguments
    ///
    /// * `batch_dims` - Batch dimensions (e.g., &[batch_size])
    /// * `feature_dims` - Feature dimensions (e.g., &[channels, height, width])
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::batched(&[32], &[3, 224, 224]);
    /// assert_eq!(s.ndim(), 4);
    /// assert_eq!(s.as_slice(), &[32, 3, 224, 224]);
    /// ```
    pub fn batched(batch_dims: &[usize], feature_dims: &[usize]) -> Self {
        let mut dims = batch_dims.to_vec();
        dims.extend_from_slice(feature_dims);
        Shape(dims)
    }

    /// Creates a shape from a flat index and a shape, computing the
    /// corresponding multi-dimensional indices (inverse of compute_index).
    ///
    /// This is used for converting flat array indices to multi-dimensional
    /// coordinates.
    ///
    /// # Arguments
    ///
    /// * `flat_index` - The flat (linear) index
    /// * `shape` - The shape to index into
    ///
    /// # Returns
    ///
    /// A vector of indices, one per dimension.
    pub fn from_flat_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0usize; shape.len()];
        let mut remaining = flat_index;
        for i in (0..shape.len()).rev() {
            if shape[i] > 0 {
                indices[i] = remaining % shape[i];
                remaining /= shape[i];
            }
        }
        indices
    }
}

// =============================================================================
// Shape Core Methods
// =============================================================================

impl Shape {
    /// Returns the number of dimensions (rank) of this shape.
    ///
    /// A scalar has rank 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert_eq!(Shape::scalar().ndim(), 0);
    /// assert_eq!(Shape::vector(5).ndim(), 1);
    /// assert_eq!(Shape::matrix(3, 4).ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Returns the total number of elements in this shape.
    ///
    /// This is the product of all dimension sizes. For a scalar, returns 1.
    /// For an empty shape (0 dimensions), returns 1.
    ///
    /// # Panics
    ///
    /// Does not panic even with empty shapes.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert_eq!(Shape::scalar().numel(), 1);
    /// assert_eq!(Shape::from_dims(&[2, 3, 4]).numel(), 24);
    /// assert_eq!(Shape::from_dims(&[0, 5]).numel(), 0);
    /// ```
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Returns the shape dimensions as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3]);
    /// assert_eq!(s.as_slice(), &[2, 3]);
    /// ```
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Returns the shape dimensions as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let mut s = Shape::from_dims(&[2, 3]);
    /// s.to_mut()[0] = 5;
    /// assert_eq!(s.size_at(0), 5);
    /// ```
    pub fn to_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }

    /// Returns a clone of the underlying dimension vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3]);
    /// let v = s.to_vec();
    /// assert_eq!(v, vec![2, 3]);
    /// ```
    pub fn to_vec(&self) -> Vec<usize> {
        self.0.clone()
    }

    /// Returns the size of the dimension at the given axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The dimension index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `axis` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// assert_eq!(s.size_at(0), 2);
    /// assert_eq!(s.size_at(1), 3);
    /// assert_eq!(s.size_at(2), 4);
    /// ```
    pub fn size_at(&self, axis: usize) -> usize {
        self.0[axis]
    }

    /// Sets the size of the dimension at the given axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The dimension index
    /// * `size` - The new size
    ///
    /// # Panics
    ///
    /// Panics if `axis` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let mut s = Shape::from_dims(&[2, 3, 4]);
    /// s.set_size(1, 10);
    /// assert_eq!(s.size_at(1), 10);
    /// ```
    pub fn set_size(&mut self, axis: usize, size: usize) {
        self.0[axis] = size;
    }

    /// Returns whether the given axis is in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3]);
    /// assert!(s.has_axis(0));
    /// assert!(s.has_axis(1));
    /// assert!(!s.has_axis(2));
    /// ```
    pub fn has_axis(&self, axis: usize) -> bool {
        axis < self.0.len()
    }

    /// Returns the size of the last dimension, or 1 for scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert_eq!(Shape::from_dims(&[2, 3]).last_dim(), 3);
    /// assert_eq!(Shape::scalar().last_dim(), 1);
    /// ```
    pub fn last_dim(&self) -> usize {
        self.0.last().copied().unwrap_or(1)
    }

    /// Returns the size of the first dimension, or 1 for scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert_eq!(Shape::from_dims(&[2, 3]).first_dim(), 2);
    /// assert_eq!(Shape::scalar().first_dim(), 1);
    /// ```
    pub fn first_dim(&self) -> usize {
        self.0.first().copied().unwrap_or(1)
    }

    /// Returns an iterator over the dimension sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// let sizes: Vec<usize> = s.iter().copied().collect();
    /// assert_eq!(sizes, vec![2, 3, 4]);
    /// ```
    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.0.iter()
    }

    /// Returns a mutable iterator over the dimension sizes.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, usize> {
        self.0.iter_mut()
    }

    /// Returns whether this shape has zero elements (any dimension is 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert!(!Shape::from_dims(&[2, 3]).is_empty());
    /// assert!(Shape::from_dims(&[0, 3]).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0.iter().any(|&d| d == 0)
    }

    /// Returns whether this shape is a scalar (0 dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert!(Shape::scalar().is_scalar());
    /// assert!(!Shape::vector(5).is_scalar());
    /// ```
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns whether this shape represents a 1-dimensional tensor (vector).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert!(Shape::vector(5).is_vector());
    /// assert!(!Shape::matrix(3, 4).is_vector());
    /// ```
    pub fn is_vector(&self) -> bool {
        self.0.len() == 1
    }

    /// Returns whether this shape represents a 2-dimensional tensor (matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert!(Shape::matrix(3, 4).is_matrix());
    /// assert!(!Shape::vector(5).is_matrix());
    /// ```
    pub fn is_matrix(&self) -> bool {
        self.0.len() == 2
    }

    /// Computes the flat (linear) index for the given multi-dimensional indices.
    ///
    /// This assumes row-major (C-style) memory layout.
    ///
    /// # Arguments
    ///
    /// * `indices` - Multi-dimensional indices
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3]);
    /// assert_eq!(s.compute_index(&[0, 0]), 0);
    /// assert_eq!(s.compute_index(&[0, 1]), 1);
    /// assert_eq!(s.compute_index(&[1, 0]), 3);
    /// assert_eq!(s.compute_index(&[1, 2]), 5);
    /// ```
    pub fn compute_index(&self, indices: &[usize]) -> usize {
        let strides = self.row_major_strides();
        indices
            .iter()
            .zip(strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Returns the position of the first dimension of size 1, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert_eq!(Shape::from_dims(&[1, 3, 4]).first_size_one_dim(), Some(0));
    /// assert_eq!(Shape::from_dims(&[2, 1, 4]).first_size_one_dim(), Some(1));
    /// assert_eq!(Shape::from_dims(&[2, 3, 4]).first_size_one_dim(), None);
    /// ```
    pub fn first_size_one_dim(&self) -> Option<usize> {
        self.0.iter().position(|&d| d == 1)
    }

    /// Returns the number of dimensions with size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// assert_eq!(Shape::from_dims(&[1, 3, 1, 4]).num_size_one_dims(), 2);
    /// assert_eq!(Shape::from_dims(&[2, 3, 4]).num_size_one_dims(), 0);
    /// ```
    pub fn num_size_one_dims(&self) -> usize {
        self.0.iter().filter(|&&d| d == 1).count()
    }

    /// Returns the product of all dimensions except the given axis.
    ///
    /// This is useful for computing batch sizes or feature counts.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// assert_eq!(s.product_except(0), 12);
    /// assert_eq!(s.product_except(1), 8);
    /// assert_eq!(s.product_except(2), 6);
    /// ```
    pub fn product_except(&self, axis: usize) -> usize {
        self.0
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .product()
    }
}

// =============================================================================
// Shape Manipulation Methods
// =============================================================================

impl Shape {
    /// Returns a new shape with an additional dimension of size 1 inserted
    /// at the given axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - Position to insert the new dimension
    ///
    /// # Panics
    ///
    /// Panics if `axis` > ndim.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[3, 4]);
    /// let expanded = s.insert_dim(0);
    /// assert_eq!(expanded.as_slice(), &[1, 3, 4]);
    /// let expanded2 = s.insert_dim(2);
    /// assert_eq!(expanded2.as_slice(), &[3, 4, 1]);
    /// ```
    pub fn insert_dim(&self, axis: usize) -> Shape {
        assert!(axis <= self.0.len(), "axis {} out of bounds for ndim {}", axis, self.0.len());
        let mut dims = self.0.clone();
        dims.insert(axis, 1);
        Shape(dims)
    }

    /// Returns a new shape with the dimension at the given axis removed.
    ///
    /// The dimension must have size 1.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove
    ///
    /// # Panics
    ///
    /// Panics if the axis is out of bounds or the dimension is not size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[1, 3, 1, 4]);
    /// let squeezed = s.remove_dim(0);
    /// assert_eq!(squeezed.as_slice(), &[3, 1, 4]);
    /// ```
    pub fn remove_dim(&self, axis: usize) -> Shape {
        assert!(axis < self.0.len(), "axis {} out of bounds for ndim {}", axis, self.0.len());
        assert_eq!(self.0[axis], 1, "cannot remove dimension {} with size {}", axis, self.0[axis]);
        let mut dims = self.0.clone();
        dims.remove(axis);
        Shape(dims)
    }

    /// Returns a new shape with the given axis set to size 1.
    ///
    /// This is the inverse of `expand_shape` for size-1 dimensions.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to set to size 1
    ///
    /// # Panics
    ///
    /// Panics if `axis` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[3, 4, 5]);
    /// let with_dim = s.with_dim(1, 1);
    /// assert_eq!(with_dim.as_slice(), &[3, 1, 5]);
    /// ```
    pub fn with_dim(&self, axis: usize, size: usize) -> Shape {
        let mut dims = self.0.clone();
        dims[axis] = size;
        Shape(dims)
    }

    /// Returns a new shape without the given axis.
    ///
    /// Unlike `remove_dim`, this does not require the dimension to be size 1.
    /// This is equivalent to reducing the tensor along that axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove
    ///
    /// # Panics
    ///
    /// Panics if `axis` is out of bounds.
    pub fn without_dim(&self, axis: usize) -> Shape {
        assert!(axis < self.0.len(), "axis {} out of bounds for ndim {}", axis, self.0.len());
        let mut dims = self.0.clone();
        dims.remove(axis);
        Shape(dims)
    }

    /// Squeezes all dimensions of size 1 from the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[1, 3, 1, 4, 1]);
    /// let squeezed = s.squeeze();
    /// assert_eq!(squeezed.as_slice(), &[3, 4]);
    /// ```
    pub fn squeeze(&self) -> Shape {
        let dims: Vec<usize> = self.0.iter().filter(|&&d| d != 1).copied().collect();
        Shape(dims)
    }

    /// Squeezes specific dimensions of size 1 from the shape.
    ///
    /// Dimensions that are not size 1 are left unchanged.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes to squeeze
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[1, 3, 1, 4, 1]);
    /// let squeezed = s.squeeze_shape(&[0, 2]);
    /// assert_eq!(squeezed.as_slice(), &[3, 4, 1]);
    /// ```
    pub fn squeeze_shape(&self, axes: &[usize]) -> Shape {
        let mut dims = self.0.clone();
        // Sort axes in reverse to remove from highest index first
        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort();
        sorted_axes.reverse();
        for &axis in &sorted_axes {
            if axis < dims.len() && dims[axis] == 1 {
                dims.remove(axis);
            }
        }
        Shape(dims)
    }

    /// Expands the shape by broadcasting to the given target shape.
    ///
    /// Dimensions of size 1 can be expanded to any size.
    /// Other dimensions must match the target shape exactly.
    ///
    /// # Arguments
    ///
    /// * `target` - The target shape to broadcast to
    ///
    /// # Panics
    ///
    /// Panics if broadcasting is not possible.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[1, 3, 1]);
    /// let expanded = s.broadcast_shape(&[4, 3, 5]);
    /// assert_eq!(expanded.as_slice(), &[4, 3, 5]);
    /// ```
    pub fn broadcast_shape(&self, target: &[usize]) -> Shape {
        Shape(broadcast_shapes_impl(self.as_slice(), target))
    }

    /// Expands a dimension by the given factor.
    ///
    /// The dimension at the given axis must have size 1.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to expand
    /// * `size` - The new size (must be > 1)
    ///
    /// # Panics
    ///
    /// Panics if the dimension is not size 1.
    pub fn expand_dim(&self, axis: usize, size: usize) -> Shape {
        assert_eq!(self.0[axis], 1, "cannot expand dimension {} with size {}", axis, self.0[axis]);
        self.with_dim(axis, size)
    }

    /// Reverses the dimensions of this shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// assert_eq!(s.reversed().as_slice(), &[4, 3, 2]);
    /// ```
    pub fn reversed(&self) -> Shape {
        let dims: Vec<usize> = self.0.iter().rev().copied().collect();
        Shape(dims)
    }

    /// Permutes the dimensions according to the given permutation.
    ///
    /// # Arguments
    ///
    /// * `permutation` - A permutation of [0, ndim)
    ///
    /// # Panics
    ///
    /// Panics if the permutation is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// let permuted = s.permuted(&[2, 0, 1]);
    /// assert_eq!(permuted.as_slice(), &[4, 2, 3]);
    /// ```
    pub fn permuted(&self, permutation: &[usize]) -> Shape {
        assert_eq!(permutation.len(), self.0.len(), "permutation length mismatch");
        let dims: Vec<usize> = permutation.iter().map(|&i| self.0[i]).collect();
        Shape(dims)
    }

    /// Returns the inverse permutation of the given permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// let inv = s.inverse_perm(&[2, 0, 1]);
    /// assert_eq!(inv, vec![1, 2, 0]);
    /// ```
    pub fn inverse_perm(&self, permutation: &[usize]) -> Vec<usize> {
        let mut inv = vec![0usize; permutation.len()];
        for (i, &p) in permutation.iter().enumerate() {
            inv[p] = i;
        }
        inv
    }

    /// Pads the shape with leading size-1 dimensions to reach the target rank.
    ///
    /// # Arguments
    ///
    /// * `target_rank` - The minimum number of dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[3, 4]);
    /// let padded = s.pad_to_rank(4);
    /// assert_eq!(padded.as_slice(), &[1, 1, 3, 4]);
    /// ```
    pub fn pad_to_rank(&self, target_rank: usize) -> Shape {
        if self.0.len() >= target_rank {
            return self.clone();
        }
        let padding = target_rank - self.0.len();
        let mut dims = vec![1usize; padding];
        dims.extend_from_slice(&self.0);
        Shape(dims)
    }

    /// Removes trailing dimensions of size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[3, 4, 1, 1]);
    /// assert_eq!(s.trim_trailing_ones().as_slice(), &[3, 4]);
    /// ```
    pub fn trim_trailing_ones(&self) -> Shape {
        let mut dims = self.0.clone();
        while dims.last() == Some(&1) {
            dims.pop();
        }
        Shape(dims)
    }

    /// Merges two shapes by concatenating their dimensions.
    pub fn merge(s1: &Shape, s2: &Shape) -> Shape {
        let mut dims = Vec::with_capacity(s1.ndim() + s2.ndim());
        dims.extend_from_slice(s1.as_slice());
        dims.extend_from_slice(s2.as_slice());
        Shape(dims)
    }

    /// Splits this shape into two shapes at the specified dimension index.
    pub fn split(&self, axis: usize) -> (Shape, Shape) {
        assert!(axis <= self.ndim(), "split axis {} exceeds ndim {}", axis, self.ndim());
        let left = Shape::from_dims(&self.0[..axis]);
        let right = Shape::from_dims(&self.0[axis..]);
        (left, right)
    }

    /// Transposes two dimensions of the shape.
    pub fn transposed(&self, dim0: usize, dim1: usize) -> Shape {
        assert!(dim0 < self.ndim(), "dim0 {} out of bounds for ndim {}", dim0, self.ndim());
        assert!(dim1 < self.ndim(), "dim1 {} out of bounds for ndim {}", dim1, self.ndim());
        let mut dims = self.0.clone();
        dims.swap(dim0, dim1);
        Shape(dims)
    }

    /// Checks if this shape can be broadcast with another shape.
    pub fn is_broadcastable_with(&self, other: &Shape) -> bool {
        let n1 = self.ndim();
        let n2 = other.ndim();
        let max_n = n1.max(n2);
        for i in 0..max_n {
            let d1 = if i < n1 { self.0[n1 - 1 - i] } else { 1 };
            let d2 = if i < n2 { other.0[n2 - 1 - i] } else { 1 };
            if d1 != d2 && d1 != 1 && d2 != 1 {
                return false;
            }
        }
        true
    }

    /// Broadcasts this shape to a target shape, returning an error if incompatible.
    pub fn broadcast_to(&self, target: &[usize]) -> BrainResult<Shape> {
        let n_src = self.ndim();
        let n_tgt = target.len();
        if n_src > n_tgt {
            return Err(BrainError::shape_mismatch(
                format!("{:?}", target),
                format!("{:?}", self.0),
                "broadcast_to: cannot broadcast to lower rank",
            ));
        }
        let mut result = target.to_vec();
        for i in 0..n_src {
            let src_dim = self.0[n_src - 1 - i];
            let tgt_dim = target[n_tgt - 1 - i];
            if src_dim != tgt_dim && src_dim != 1 {
                return Err(BrainError::shape_mismatch(
                    format!("{:?}", target),
                    format!("{:?}", self.0),
                    format!("broadcast_to: dimension mismatch at trailing index {}", i),
                ));
            }
        }
        Ok(Shape(result))
    }

    /// Broadcasts multiple shapes together to compute their common broadcast shape.
    pub fn broadcast_shapes(shapes: &[&Shape]) -> BrainResult<Shape> {
        if shapes.is_empty() {
            return Ok(Shape::scalar());
        }
        let max_rank = shapes.iter().map(|s| s.ndim()).max().unwrap_or(0);
        let mut result_dims = vec![1usize; max_rank];

        for shape in shapes {
            let ndim = shape.ndim();
            for i in 0..ndim {
                let dim = shape.0[ndim - 1 - i];
                let out_idx = max_rank - 1 - i;
                let cur = result_dims[out_idx];
                if cur == 1 {
                    result_dims[out_idx] = dim;
                } else if dim != 1 && dim != cur {
                    return Err(BrainError::shape_mismatch(
                        format!("compatible with dim {}", cur),
                        format!("dim {}", dim),
                        format!("broadcast_shapes: incompatible dimension at trailing position {}", i),
                    ));
                }
            }
        }
        Ok(Shape(result_dims))
    }

    /// Sliding-window output dimension: `(in + 2*pad - dilation*(kernel-1) - 1) / stride + 1`.
    ///
    /// Computed in `i128` so a kernel larger than the padded input cannot
    /// underflow/overflow `usize` and trigger an OOM or an opaque panic. When
    /// the configuration is degenerate (kernel larger than the padded input, or a
    /// zero stride) this returns `0`, yielding an empty output tensor instead of
    /// crashing.
    pub fn output_dim(in_len: usize, padding: usize, kernel: usize, stride: usize, dilation: usize) -> usize {
        let num = (in_len as i128) + 2 * (padding as i128) - (dilation as i128) * (kernel as i128 - 1) - 1;
        if num < 0 || stride == 0 {
            return 0;
        }
        (num / stride as i128 + 1) as usize
    }

    /// Narrows a dimension to a sub-range `[start, start + length)`.
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> BrainResult<Shape> {
        if axis >= self.ndim() {
            return Err(BrainError::index_out_of_bounds(
                axis as isize,
                self.ndim(),
                Some(axis),
                "narrow: axis out of bounds",
            ));
        }
        let dim_size = self.0[axis];
        if start + length > dim_size {
            return Err(BrainError::invalid_value(format!(
                "narrow: start ({}) + length ({}) > dim_size ({}) along axis {}",
                start, length, dim_size, axis
            )));
        }
        let mut new_dims = self.0.clone();
        new_dims[axis] = length;
        Ok(Shape(new_dims))
    }

    /// Expands singleton dimensions to specified target dimensions.
    pub fn expanded(&self, target_dims: &[usize]) -> BrainResult<Shape> {
        if self.ndim() != target_dims.len() {
            return Err(BrainError::shape_mismatch(
                format!("{:?}", target_dims),
                format!("{:?}", self.0),
                "expanded: rank mismatch",
            ));
        }
        let mut result = Vec::with_capacity(target_dims.len());
        for (i, (&src, &tgt)) in self.0.iter().zip(target_dims.iter()).enumerate() {
            if src == tgt || src == 1 {
                result.push(tgt);
            } else {
                return Err(BrainError::shape_mismatch(
                    format!("dim[{}] == 1 or {}", i, tgt),
                    format!("dim[{}] == {}", i, src),
                    "expanded: cannot expand non-singleton dimension",
                ));
            }
        }
        Ok(Shape(result))
    }

    /// Validates the shape invariants.
    pub fn validate(&self) -> BrainResult<()> {
        for (axis, &dim) in self.0.iter().enumerate() {
            if dim == 0 && self.numel() != 0 {
                return Err(BrainError::invalid_value(format!(
                    "Shape has zero dimension at axis {} with non-zero numel",
                    axis
                )));
            }
        }
        Ok(())
    }

    /// Validates multi-dimensional indices against this shape.
    pub fn validate_index(&self, indices: &[usize]) -> BrainResult<()> {
        if indices.len() != self.ndim() {
            return Err(BrainError::shape_mismatch(
                format!("rank {}", self.ndim()),
                format!("rank {}", indices.len()),
                "validate_index: index rank mismatch",
            ));
        }
        for (axis, (&idx, &dim)) in indices.iter().zip(self.0.iter()).enumerate() {
            if idx >= dim {
                return Err(BrainError::index_out_of_bounds(
                    idx as isize,
                    dim,
                    Some(axis),
                    "validate_index",
                ));
            }
        }
        Ok(())
    }

    /// Validates strides for this shape.
    pub fn validate_strides(&self, strides: &[usize]) -> BrainResult<()> {
        if strides.len() != self.ndim() {
            return Err(BrainError::shape_mismatch(
                format!("strides len {}", self.ndim()),
                format!("strides len {}", strides.len()),
                "validate_strides: strides rank mismatch",
            ));
        }
        Ok(())
    }

    /// Validates that a permutation is a valid permutation of `[0, ndim)`.
    pub fn validate_permutation(&self, perm: &[usize]) -> BrainResult<()> {
        if perm.len() != self.ndim() {
            return Err(BrainError::shape_mismatch(
                format!("permutation len {}", self.ndim()),
                format!("permutation len {}", perm.len()),
                "validate_permutation: permutation length mismatch",
            ));
        }
        let mut seen = vec![false; self.ndim()];
        for &axis in perm {
            if axis >= self.ndim() {
                return Err(BrainError::index_out_of_bounds(
                    axis as isize,
                    self.ndim(),
                    None,
                    "validate_permutation: axis out of bounds",
                ));
            }
            if seen[axis] {
                return Err(BrainError::invalid_value(format!(
                    "validate_permutation: duplicate axis {} in permutation",
                    axis
                )));
            }
            seen[axis] = true;
        }
        Ok(())
    }
}

// =============================================================================
// Stride Methods
// =============================================================================

impl Shape {
    /// Computes the row-major (C-style) strides for this shape.
    ///
    /// In row-major order, the last dimension varies fastest.
    /// The stride of dimension i is the product of all dimensions after i.
    ///
    /// # Returns
    ///
    /// A vector of strides, one per dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// assert_eq!(s.row_major_strides(), vec![12, 4, 1]);
    /// ```
    pub fn row_major_strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.0.len()];
        if !self.0.is_empty() {
            strides[self.0.len() - 1] = 1;
            for i in (0..self.0.len() - 1).rev() {
                strides[i] = strides[i + 1] * self.0[i + 1];
            }
        }
        strides
    }

    /// Computes the column-major (Fortran-style) strides for this shape.
    ///
    /// In column-major order, the first dimension varies fastest.
    /// The stride of dimension i is the product of all dimensions before i.
    ///
    /// # Returns
    ///
    /// A vector of strides, one per dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// assert_eq!(s.col_major_strides(), vec![1, 2, 6]);
    /// ```
    pub fn col_major_strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.0.len()];
        if !self.0.is_empty() {
            strides[0] = 1;
            for i in 1..self.0.len() {
                strides[i] = strides[i - 1] * self.0[i - 1];
            }
        }
        strides
    }

    /// Checks if this shape with the given row-major strides is contiguous.
    ///
    /// A tensor is contiguous in row-major order if the strides match
    /// the expected row-major strides for its shape.
    ///
    /// # Arguments
    ///
    /// * `strides` - The actual strides of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// let strides = s.row_major_strides();
    /// assert!(s.is_contiguous(&strides));
    /// ```
    pub fn is_contiguous(&self, strides: &[usize]) -> bool {
        if self.0.is_empty() {
            return true;
        }
        let expected = self.row_major_strides();
        strides == expected
    }

    /// Checks if this shape with the given strides is contiguous in
    /// column-major (Fortran) order.
    ///
    /// # Arguments
    ///
    /// * `strides` - The actual strides of the tensor
    pub fn is_fortran_contiguous(&self, strides: &[usize]) -> bool {
        if self.0.is_empty() {
            return true;
        }
        let expected = self.col_major_strides();
        strides == expected
    }

    /// Returns stride info for this shape in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::shape::Shape;
    /// let s = Shape::from_dims(&[2, 3, 4]);
    /// let info = s.stride_info();
    /// assert_eq!(info.strides(), &[12, 4, 1]);
    /// ```
    pub fn stride_info(&self) -> StrideInfo {
        let strides = self.row_major_strides();
        StrideInfo {
            shape: self.0.clone(),
            strides,
        }
    }

    /// Returns stride info for this shape in column-major order.
    pub fn fortran_stride_info(&self) -> StrideInfo {
        let strides = self.col_major_strides();
        StrideInfo {
            shape: self.0.clone(),
            strides,
        }
    }
}

// =============================================================================
// Trait Implementations for Shape
// =============================================================================

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Shape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &usize {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut usize {
        &mut self.0[index]
    }
}

impl Default for Shape {
    /// Returns a scalar shape as the default.
    fn default() -> Self {
        Shape::scalar()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape(dims)
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

impl<'a> From<&'a [usize]> for Shape {
    fn from(dims: &'a [usize]) -> Self {
        Shape(dims.to_vec())
    }
}

// =============================================================================
// ShapeIndex - Multi-dimensional indexing
// =============================================================================

/// A multi-dimensional index used for advanced tensor indexing.
///
/// `ShapeIndex` provides a flexible way to specify indices across multiple
/// dimensions, supporting direct indices, ranges, and full slices.
///
/// # Index Types
///
/// * `Index(i)` - A single element index
/// * `Slice(start, end, step)` - A range with step
/// * `All` - Select all elements along this dimension
///
/// # Examples
///
/// ```
/// use brain_core::shape::{Shape, ShapeIndex, ShapeIndexType};
/// let idx = ShapeIndex::from_vec(vec![
///     ShapeIndexType::Index(0),
///     ShapeIndexType::Slice { start: Some(1), end: Some(5), step: Some(2) },
///     ShapeIndexType::All,
/// ]);
/// assert_eq!(idx.ndim(), 3);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeIndex(Vec<ShapeIndexType>);

/// The type of index for a single dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeIndexType {
    /// A single element index.
    Index(isize),
    /// A range slice with optional start, end, and step.
    /// None for start means beginning, None for end means end.
    /// Step must be > 0 if provided.
    Slice {
        start: Option<isize>,
        end: Option<isize>,
        step: Option<isize>,
    },
    /// Select all elements along this dimension (equivalent to `Slice(None, None, None)`).
    All,
    /// A new axis insertion (adds a dimension of size 1 at this position).
    NewAxis,
}

impl ShapeIndex {
    /// Creates a new ShapeIndex from a vector of index types.
    pub fn new(indices: Vec<ShapeIndexType>) -> Self {
        ShapeIndex(indices)
    }

    /// Creates a ShapeIndex from a vector of index types.
    pub fn from_vec(indices: Vec<ShapeIndexType>) -> Self {
        ShapeIndex(indices)
    }

    /// Creates a ShapeIndex that selects all elements (e.g., `tensor[...]`).
    ///
    /// # Arguments
    ///
    /// * `ndim` - Number of dimensions
    pub fn all(ndim: usize) -> Self {
        ShapeIndex(vec![ShapeIndexType::All; ndim])
    }

    /// Returns the number of index components.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Returns the index type at the given position.
    pub fn get(&self, index: usize) -> Option<&ShapeIndexType> {
        self.0.get(index)
    }

    /// Returns the index types as a slice.
    pub fn as_slice(&self) -> &[ShapeIndexType] {
        &self.0
    }

    /// Computes the output shape resulting from applying this index to a shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the tensor being indexed
    ///
    /// # Returns
    ///
    /// The output shape after indexing.
    pub fn output_shape(&self, input_shape: &Shape) -> Shape {
        let mut output_dims = Vec::new();
        let mut dim_idx = 0;

        for idx_type in &self.0 {
            match idx_type {
                ShapeIndexType::Index(_) => {
                    // Single index reduces this dimension
                    dim_idx += 1;
                }
                ShapeIndexType::Slice { start, end, step } => {
                    let dim_size = input_shape.size_at(dim_idx);
                    let resolved_start = start.unwrap_or(0).max(0) as usize;
                    let resolved_end = end.unwrap_or(dim_size as isize).max(0) as usize;
                    let resolved_step = step.unwrap_or(1).max(1) as usize;
                    let output_size = if resolved_end <= resolved_start {
                        0
                    } else {
                        (resolved_end - resolved_start + resolved_step - 1) / resolved_step
                    };
                    output_dims.push(output_size);
                    dim_idx += 1;
                }
                ShapeIndexType::All => {
                    output_dims.push(input_shape.size_at(dim_idx));
                    dim_idx += 1;
                }
                ShapeIndexType::NewAxis => {
                    output_dims.push(1);
                    // Don't increment dim_idx - NewAxis doesn't consume a dimension
                }
            }
        }

        // Include any remaining dimensions not covered by the index
        while dim_idx < input_shape.ndim() {
            output_dims.push(input_shape.size_at(dim_idx));
            dim_idx += 1;
        }

        Shape::new(output_dims)
    }

    /// Returns the number of NewAxis insertions in this index.
    pub fn num_new_axes(&self) -> usize {
        self.0.iter().filter(|t| matches!(t, ShapeIndexType::NewAxis)).count()
    }

    /// Returns the number of dimensions consumed by this index.
    ///
    /// This excludes NewAxis entries.
    pub fn consumed_dims(&self) -> usize {
        self.0.iter().filter(|t| !matches!(t, ShapeIndexType::NewAxis)).count()
    }
}

// =============================================================================
// StrideInfo
// =============================================================================

/// Stores shape and stride information for a tensor.
///
/// `StrideInfo` combines a shape with its corresponding strides,
/// enabling efficient index computation and contiguity checks.
///
/// # Examples
///
/// ```
/// use brain_core::shape::Shape;
/// let s = Shape::from_dims(&[2, 3, 4]);
/// let info = s.stride_info();
/// assert_eq!(info.shape(), &[2, 3, 4]);
/// assert_eq!(info.strides(), &[12, 4, 1]);
/// assert_eq!(info.numel(), 24);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StrideInfo {
    /// The shape dimensions.
    shape: Vec<usize>,
    /// The strides for each dimension.
    strides: Vec<usize>,
}

impl StrideInfo {
    /// Creates a new StrideInfo from shape and strides.
    ///
    /// # Panics
    ///
    /// Panics if shape and strides have different lengths.
    pub fn new(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        assert_eq!(shape.len(), strides.len(), "shape and strides must have same length");
        StrideInfo { shape, strides }
    }

    /// Returns the shape dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Computes the flat index for the given multi-dimensional indices.
    pub fn compute_index(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Checks if the tensor is contiguous in row-major order.
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_row_major_strides(&self.shape);
        self.strides == expected
    }

    /// Checks if the tensor is contiguous in column-major order.
    pub fn is_fortran_contiguous(&self) -> bool {
        let expected = compute_col_major_strides(&self.shape);
        self.strides == expected
    }

    /// Returns the stride at the given dimension.
    pub fn stride_at(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Returns the size at the given dimension.
    pub fn size_at(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    /// Computes the memory offset range [min, max) in bytes for this
    /// stride info given an element size.
    ///
    /// # Arguments
    ///
    /// * `element_size` - Size of each element in bytes
    pub fn memory_range(&self, element_size: usize) -> (usize, usize) {
        if self.shape.is_empty() {
            return (0, element_size);
        }
        let max_offset = self.compute_index(
            &self.shape.iter().map(|&d| if d > 0 { d - 1 } else { 0 }).collect::<Vec<_>>(),
        );
        (0, (max_offset + 1) * element_size)
    }

    /// Checks if two StrideInfo objects describe the same memory layout.
    pub fn same_layout(&self, other: &StrideInfo) -> bool {
        self.shape == other.shape && self.strides == other.strides
    }
}

impl fmt::Display for StrideInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Shape: {:?}", self.shape)?;
        writeln!(f, "Strides: {:?}", self.strides)?;
        write!(f, "Contiguous (C): {}", self.is_contiguous())?;
        Ok(())
    }
}

// =============================================================================
// Free Functions - Broadcasting
// =============================================================================

/// Broadcasts two shapes to their common output shape.
///
/// Broadcasting follows NumPy rules:
/// 1. Shapes are right-aligned
/// 2. Dimensions must be equal or one of them must be 1
/// 3. The output shape uses the larger of each dimension
///
/// # Arguments
///
/// * `a` - First shape
/// * `b` - Second shape
///
/// # Returns
///
/// The broadcast output shape.
///
/// # Panics
///
/// Panics if the shapes are not broadcast-compatible.
///
/// # Examples
///
/// ```
/// use brain_core::shape::broadcast_shapes;
/// let result = broadcast_shapes(&[2, 1], &[1, 3]);
/// assert_eq!(result, vec![2, 3]);
/// ```
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    broadcast_shapes_impl(a, b)
}

/// Internal implementation of shape broadcasting.
fn broadcast_shapes_impl(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let a_idx = a.len().wrapping_sub(max_ndim - i);
        let b_idx = b.len().wrapping_sub(max_ndim - i);

        let da = if a_idx < a.len() { a[a_idx] } else { 1 };
        let db = if b_idx < b.len() { b[b_idx] } else { 1 };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            panic!(
                "Shape mismatch in broadcasting: dimension {} has sizes {} and {}",
                i, da, db
            );
        }
    }

    result
}

/// Broadcasts multiple shapes to their common output shape.
///
/// # Arguments
///
/// * `shapes` - Slice of shapes to broadcast
///
/// # Returns
///
/// The common broadcast shape.
///
/// # Panics
///
/// Panics if the shapes are not all broadcast-compatible.
///
/// # Examples
///
/// ```
/// use brain_core::shape::broadcast_shapes_multi;
/// let result = broadcast_shapes_multi(&[&[2, 1, 4], &[1, 3, 1], &[2, 3, 4]]);
/// assert_eq!(result, vec![2, 3, 4]);
/// ```
pub fn broadcast_shapes_multi(shapes: &[&[usize]]) -> Vec<usize> {
    if shapes.is_empty() {
        return vec![];
    }
    if shapes.len() == 1 {
        return shapes[0].to_vec();
    }
    let mut result = shapes[0].to_vec();
    for shape in &shapes[1..] {
        result = broadcast_shapes_impl(&result, shape);
    }
    result
}

/// Checks if two shapes can be broadcast together without computing the result.
///
/// # Arguments
///
/// * `a` - First shape
/// * `b` - Second shape
///
/// # Returns
///
/// `true` if the shapes are broadcast-compatible.
pub fn can_broadcast(a: &[usize], b: &[usize]) -> bool {
    let max_ndim = a.len().max(b.len());
    for i in 0..max_ndim {
        let a_idx = a.len().wrapping_sub(max_ndim - i);
        let b_idx = b.len().wrapping_sub(max_ndim - i);
        let da = if a_idx < a.len() { a[a_idx] } else { 1 };
        let db = if b_idx < b.len() { b[b_idx] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return false;
        }
    }
    true
}

// =============================================================================
// Free Functions - Stride Computation
// =============================================================================

fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0usize; shape.len()];
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn compute_col_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0usize; shape.len()];
    strides[0] = 1;
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}

/// Computes row-major strides for a shape.
///
/// # Examples
///
/// ```
/// use brain_core::shape;
/// assert_eq!(shape::row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
/// assert_eq!(shape::row_major_strides(&[5]), vec![1]);
/// ```
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    compute_row_major_strides(shape)
}

/// Computes column-major strides for a shape.
///
/// # Examples
///
/// ```
/// use brain_core::shape;
/// assert_eq!(shape::col_major_strides(&[2, 3, 4]), vec![1, 2, 6]);
/// ```
pub fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    compute_col_major_strides(shape)
}

// =============================================================================
// Free Functions - Output Shape Computation
// =============================================================================

/// Computes the output shape of a 2D convolution.
///
/// # Arguments
///
/// * `input_shape` - Shape `[N, C_in, H, W]`
/// * `kernel_size` - `[kH, kW]`
/// * `stride` - `[sH, sW]`
/// * `padding` - `[pH, pW]`
/// * `dilation` - `[dH, dW]`
/// * `groups` - Number of convolution groups
///
/// # Returns
///
/// Output shape `[N, C_out, H_out, W_out]`
///
/// # Examples
///
/// ```
/// use brain_core::shape;
/// let out = shape::compute_conv_output_shape(
///     &[1, 3, 32, 32],   // input
///     &[3, 3],            // kernel
///     &[1, 1],            // stride
///     &[1, 1],            // padding
///     &[1, 1],            // dilation
///     1,                  // groups
///     16,                 // out_channels
/// );
/// assert_eq!(out, vec![1, 16, 32, 32]);
/// ```
pub fn compute_conv_output_shape(
    input_shape: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize,
    out_channels: usize,
) -> Vec<usize> {
    assert!(input_shape.len() == 4, "conv input must be 4D");
    assert!(kernel_size.len() == 2, "kernel must be 2D");

    let h_in = input_shape[2] as i64;
    let w_in = input_shape[3] as i64;
    let kh = kernel_size[0] as i64;
    let kw = kernel_size[1] as i64;
    let sh = stride[0] as i64;
    let sw = stride[1] as i64;
    let ph = padding[0] as i64;
    let pw = padding[1] as i64;
    let dh = dilation[0] as i64;
    let dw = dilation[1] as i64;

    let h_out = ((h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1) as usize;
    let w_out = ((w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1) as usize;

    vec![input_shape[0], out_channels, h_out, w_out]
}

/// Computes the output shape of a 2D pooling operation.
///
/// # Arguments
///
/// * `input_shape` - Shape `[N, C, H, W]`
/// * `kernel_size` - `[kH, kW]`
/// * `stride` - `[sH, sW]`
/// * `padding` - `[pH, pW]`
///
/// # Returns
///
/// Output shape `[N, C, H_out, W_out]`
///
/// # Examples
///
/// ```
/// use brain_core::shape;
/// let out = shape::compute_pool_output_shape(
///     &[1, 3, 32, 32],   // input
///     &[2, 2],            // kernel
///     &[2, 2],            // stride
///     &[0, 0],            // padding
/// );
/// assert_eq!(out, vec![1, 3, 16, 16]);
/// ```
pub fn compute_pool_output_shape(
    input_shape: &[usize],
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
) -> Vec<usize> {
    assert!(input_shape.len() == 4, "pool input must be 4D");
    assert!(kernel_size.len() == 2, "kernel must be 2D");

    let h_in = input_shape[2] as i64;
    let w_in = input_shape[3] as i64;
    let kh = kernel_size[0] as i64;
    let kw = kernel_size[1] as i64;
    let sh = stride[0] as i64;
    let sw = stride[1] as i64;
    let ph = padding[0] as i64;
    let pw = padding[1] as i64;

    let h_out = ((h_in + 2 * ph - kh) / sh + 1) as usize;
    let w_out = ((w_in + 2 * pw - kw) / sw + 1) as usize;

    vec![input_shape[0], input_shape[1], h_out, w_out]
}

/// Computes the output shape of a transpose/permute operation.
///
/// # Arguments
///
/// * `input_shape` - Original shape
/// * `permutation` - Permutation of axes
///
/// # Returns
///
/// Transposed shape.
///
/// # Examples
///
/// ```
/// use brain_core::shape;
/// let out = shape::compute_transpose_shape(&[2, 3, 4], &[2, 0, 1]);
/// assert_eq!(out, vec![4, 2, 3]);
/// ```
pub fn compute_transpose_shape(input_shape: &[usize], permutation: &[usize]) -> Vec<usize> {
    permutation.iter().map(|&i| input_shape[i]).collect()
}

/// Computes the output shape of a reshape operation, handling -1 dimension.
///
/// The special value -1 in the target shape means "infer this dimension
/// from the total number of elements and the other dimensions."
///
/// # Arguments
///
/// * `input_shape` - Original shape
/// * `target_shape` - Target shape (may contain -1)
///
/// # Returns
///
/// The resolved output shape with -1 replaced.
///
/// # Panics
///
/// Panics if the reshape is invalid or -1 cannot be resolved.
///
/// # Examples
///
/// ```
/// use brain_core::shape;
/// let out = shape::compute_reshape_shape(&[2, 3, 4], &[-1, 4]);
/// assert_eq!(out, vec![6, 4]);
/// ```
pub fn compute_reshape_shape(input_shape: &[usize], target_shape: &[isize]) -> Vec<usize> {
    let input_numel: usize = input_shape.iter().product();
    let mut neg_idx = None;
    let mut known_product = 1usize;

    for (i, &dim) in target_shape.iter().enumerate() {
        if dim == -1 {
            assert!(neg_idx.is_none(), "only one dimension can be -1 in reshape");
            neg_idx = Some(i);
        } else if dim < -1 {
            panic!("reshape dimension cannot be negative (got {})", dim);
        } else {
            known_product *= dim as usize;
        }
    }

    let mut result: Vec<usize> = target_shape.iter().map(|&d| d as usize).collect();

    if let Some(idx) = neg_idx {
        assert!(
            known_product > 0 && input_numel % known_product == 0,
            "cannot reshape: {} elements cannot be divided into shape with product {}",
            input_numel,
            known_product
        );
        result[idx] = input_numel / known_product;
    } else {
        assert_eq!(
            known_product, input_numel,
            "cannot reshape: input has {} elements but target shape has {}",
            input_numel, known_product
        );
    }

    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Constructor Tests
    // =========================================================================

    #[test]
    fn test_new() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_from_dims() {
        let s = Shape::from_dims(&[2, 3]);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.numel(), 6);
    }

    #[test]
    fn test_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert!(s.is_scalar());
    }

    #[test]
    fn test_vector() {
        let s = Shape::vector(10);
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.size_at(0), 10);
        assert!(s.is_vector());
    }

    #[test]
    fn test_matrix() {
        let s = Shape::matrix(3, 4);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.numel(), 12);
        assert!(s.is_matrix());
    }

    #[test]
    fn test_cube() {
        let s = Shape::cube(3, 4);
        assert_eq!(s.as_slice(), &[4, 4, 4]);
    }

    #[test]
    fn test_batched() {
        let s = Shape::batched(&[32], &[3, 224, 224]);
        assert_eq!(s.ndim(), 4);
        assert_eq!(s.as_slice(), &[32, 3, 224, 224]);
    }

    #[test]
    fn test_from_flat_index() {
        let shape = vec![2, 3];
        assert_eq!(Shape::from_flat_index(0, &shape), vec![0, 0]);
        assert_eq!(Shape::from_flat_index(1, &shape), vec![0, 1]);
        assert_eq!(Shape::from_flat_index(3, &shape), vec![1, 0]);
        assert_eq!(Shape::from_flat_index(5, &shape), vec![1, 2]);
    }

    #[test]
    fn test_from_flat_index_3d() {
        let shape = vec![2, 3, 4];
        assert_eq!(Shape::from_flat_index(0, &shape), vec![0, 0, 0]);
        assert_eq!(Shape::from_flat_index(1, &shape), vec![0, 0, 1]);
        assert_eq!(Shape::from_flat_index(4, &shape), vec![0, 1, 0]);
        assert_eq!(Shape::from_flat_index(12, &shape), vec![1, 0, 0]);
        assert_eq!(Shape::from_flat_index(23, &shape), vec![1, 2, 3]);
    }

    // =========================================================================
    // Core Method Tests
    // =========================================================================

    #[test]
    fn test_ndim() {
        assert_eq!(Shape::scalar().ndim(), 0);
        assert_eq!(Shape::vector(5).ndim(), 1);
        assert_eq!(Shape::matrix(3, 4).ndim(), 2);
        assert_eq!(Shape::from_dims(&[2, 3, 4, 5]).ndim(), 4);
    }

    #[test]
    fn test_numel() {
        assert_eq!(Shape::scalar().numel(), 1);
        assert_eq!(Shape::vector(5).numel(), 5);
        assert_eq!(Shape::matrix(3, 4).numel(), 12);
        assert_eq!(Shape::from_dims(&[2, 3, 4]).numel(), 24);
    }

    #[test]
    fn test_numel_zero_dim() {
        assert_eq!(Shape::from_dims(&[0, 5]).numel(), 0);
        assert_eq!(Shape::from_dims(&[5, 0]).numel(), 0);
    }

    #[test]
    fn test_as_slice() {
        let s = Shape::from_dims(&[2, 3]);
        assert_eq!(s.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_to_mut() {
        let mut s = Shape::from_dims(&[2, 3]);
        s.to_mut()[0] = 5;
        assert_eq!(s.size_at(0), 5);
    }

    #[test]
    fn test_to_vec() {
        let s = Shape::from_dims(&[2, 3]);
        assert_eq!(s.to_vec(), vec![2, 3]);
    }

    #[test]
    fn test_size_at() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.size_at(0), 2);
        assert_eq!(s.size_at(1), 3);
        assert_eq!(s.size_at(2), 4);
    }

    #[test]
    fn test_set_size() {
        let mut s = Shape::from_dims(&[2, 3, 4]);
        s.set_size(1, 10);
        assert_eq!(s.size_at(1), 10);
    }

    #[test]
    fn test_has_axis() {
        let s = Shape::from_dims(&[2, 3]);
        assert!(s.has_axis(0));
        assert!(s.has_axis(1));
        assert!(!s.has_axis(2));
    }

    #[test]
    fn test_last_dim() {
        assert_eq!(Shape::from_dims(&[2, 3]).last_dim(), 3);
        assert_eq!(Shape::scalar().last_dim(), 1);
    }

    #[test]
    fn test_first_dim() {
        assert_eq!(Shape::from_dims(&[2, 3]).first_dim(), 2);
        assert_eq!(Shape::scalar().first_dim(), 1);
    }

    #[test]
    fn test_iter() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let sizes: Vec<usize> = s.iter().copied().collect();
        assert_eq!(sizes, vec![2, 3, 4]);
    }

    #[test]
    fn test_is_empty() {
        assert!(!Shape::from_dims(&[2, 3]).is_empty());
        assert!(Shape::from_dims(&[0, 3]).is_empty());
        assert!(Shape::from_dims(&[2, 0]).is_empty());
    }

    #[test]
    fn test_is_scalar() {
        assert!(Shape::scalar().is_scalar());
        assert!(!Shape::vector(5).is_scalar());
    }

    #[test]
    fn test_is_vector() {
        assert!(Shape::vector(5).is_vector());
        assert!(!Shape::matrix(3, 4).is_vector());
        assert!(!Shape::scalar().is_vector());
    }

    #[test]
    fn test_is_matrix() {
        assert!(Shape::matrix(3, 4).is_matrix());
        assert!(!Shape::vector(5).is_matrix());
    }

    #[test]
    fn test_compute_index() {
        let s = Shape::from_dims(&[2, 3]);
        assert_eq!(s.compute_index(&[0, 0]), 0);
        assert_eq!(s.compute_index(&[0, 1]), 1);
        assert_eq!(s.compute_index(&[0, 2]), 2);
        assert_eq!(s.compute_index(&[1, 0]), 3);
        assert_eq!(s.compute_index(&[1, 2]), 5);
    }

    #[test]
    fn test_compute_index_3d() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.compute_index(&[0, 0, 0]), 0);
        assert_eq!(s.compute_index(&[0, 0, 1]), 1);
        assert_eq!(s.compute_index(&[0, 1, 0]), 4);
        assert_eq!(s.compute_index(&[1, 0, 0]), 12);
        assert_eq!(s.compute_index(&[1, 2, 3]), 23);
    }

    #[test]
    fn test_first_size_one_dim() {
        assert_eq!(Shape::from_dims(&[1, 3, 4]).first_size_one_dim(), Some(0));
        assert_eq!(Shape::from_dims(&[2, 1, 4]).first_size_one_dim(), Some(1));
        assert_eq!(Shape::from_dims(&[2, 3, 4]).first_size_one_dim(), None);
    }

    #[test]
    fn test_num_size_one_dims() {
        assert_eq!(Shape::from_dims(&[1, 3, 1, 4]).num_size_one_dims(), 2);
        assert_eq!(Shape::from_dims(&[2, 3, 4]).num_size_one_dims(), 0);
        assert_eq!(Shape::from_dims(&[1, 1, 1]).num_size_one_dims(), 3);
    }

    #[test]
    fn test_product_except() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.product_except(0), 12);
        assert_eq!(s.product_except(1), 8);
        assert_eq!(s.product_except(2), 6);
    }

    // =========================================================================
    // Manipulation Method Tests
    // =========================================================================

    #[test]
    fn test_insert_dim() {
        let s = Shape::from_dims(&[3, 4]);
        assert_eq!(s.insert_dim(0).as_slice(), &[1, 3, 4]);
        assert_eq!(s.insert_dim(1).as_slice(), &[3, 1, 4]);
        assert_eq!(s.insert_dim(2).as_slice(), &[3, 4, 1]);
    }

    #[test]
    fn test_remove_dim() {
        let s = Shape::from_dims(&[1, 3, 1, 4]);
        assert_eq!(s.remove_dim(0).as_slice(), &[3, 1, 4]);
        assert_eq!(s.remove_dim(2).as_slice(), &[1, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_remove_dim_not_size_one() {
        let s = Shape::from_dims(&[3, 4]);
        s.remove_dim(0); // size is 3, not 1
    }

    #[test]
    fn test_with_dim() {
        let s = Shape::from_dims(&[3, 4, 5]);
        assert_eq!(s.with_dim(1, 1).as_slice(), &[3, 1, 5]);
    }

    #[test]
    fn test_without_dim() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.without_dim(0).as_slice(), &[3, 4]);
        assert_eq!(s.without_dim(1).as_slice(), &[2, 4]);
    }

    #[test]
    fn test_squeeze() {
        let s = Shape::from_dims(&[1, 3, 1, 4, 1]);
        assert_eq!(s.squeeze().as_slice(), &[3, 4]);
    }

    #[test]
    fn test_squeeze_no_ones() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.squeeze().as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_squeeze_all_ones() {
        let s = Shape::from_dims(&[1, 1, 1]);
        assert_eq!(s.squeeze().ndim(), 0);
    }

    #[test]
    fn test_squeeze_shape() {
        let s = Shape::from_dims(&[1, 3, 1, 4, 1]);
        assert_eq!(s.squeeze_shape(&[0, 2]).as_slice(), &[3, 4, 1]);
        assert_eq!(s.squeeze_shape(&[0, 4]).as_slice(), &[3, 1, 4]);
    }

    #[test]
    fn test_broadcast_shape() {
        let s = Shape::from_dims(&[1, 3, 1]);
        let result = s.broadcast_shape(&[4, 3, 5]);
        assert_eq!(result.as_slice(), &[4, 3, 5]);
    }

    #[test]
    fn test_broadcast_shape_same() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let result = s.broadcast_shape(&[2, 3, 4]);
        assert_eq!(result.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_expand_dim() {
        let s = Shape::from_dims(&[1, 3, 1]);
        assert_eq!(s.expand_dim(0, 5).as_slice(), &[5, 3, 1]);
        assert_eq!(s.expand_dim(2, 7).as_slice(), &[1, 3, 7]);
    }

    #[test]
    #[should_panic]
    fn test_expand_dim_not_one() {
        let s = Shape::from_dims(&[3, 4]);
        s.expand_dim(0, 5); // dim 0 is 3, not 1
    }

    #[test]
    fn test_reversed() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.reversed().as_slice(), &[4, 3, 2]);
    }

    #[test]
    fn test_permuted() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.permuted(&[2, 0, 1]).as_slice(), &[4, 2, 3]);
        assert_eq!(s.permuted(&[1, 2, 0]).as_slice(), &[3, 4, 2]);
    }

    #[test]
    fn test_inverse_perm() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.inverse_perm(&[2, 0, 1]), vec![1, 2, 0]);
        assert_eq!(s.inverse_perm(&[0, 1, 2]), vec![0, 1, 2]);
    }

    #[test]
    fn test_pad_to_rank() {
        let s = Shape::from_dims(&[3, 4]);
        assert_eq!(s.pad_to_rank(4).as_slice(), &[1, 1, 3, 4]);
        assert_eq!(s.pad_to_rank(2).as_slice(), &[3, 4]);
        assert_eq!(s.pad_to_rank(1).as_slice(), &[3, 4]); // won't truncate
    }

    #[test]
    fn test_trim_trailing_ones() {
        let s = Shape::from_dims(&[3, 4, 1, 1]);
        assert_eq!(s.trim_trailing_ones().as_slice(), &[3, 4]);
    }

    #[test]
    fn test_trim_trailing_ones_no_trailing() {
        let s = Shape::from_dims(&[1, 3, 4]);
        assert_eq!(s.trim_trailing_ones().as_slice(), &[1, 3, 4]);
    }

    // =========================================================================
    // Stride Tests
    // =========================================================================

    #[test]
    fn test_row_major_strides() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.row_major_strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_row_major_strides_scalar() {
        let s = Shape::scalar();
        assert!(s.row_major_strides().is_empty());
    }

    #[test]
    fn test_row_major_strides_1d() {
        let s = Shape::vector(5);
        assert_eq!(s.row_major_strides(), vec![1]);
    }

    #[test]
    fn test_col_major_strides() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.col_major_strides(), vec![1, 2, 6]);
    }

    #[test]
    fn test_col_major_strides_scalar() {
        let s = Shape::scalar();
        assert!(s.col_major_strides().is_empty());
    }

    #[test]
    fn test_is_contiguous() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let strides = s.row_major_strides();
        assert!(s.is_contiguous(&strides));
    }

    #[test]
    fn test_is_not_contiguous() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert!(!s.is_contiguous(&[4, 1, 3]));
    }

    #[test]
    fn test_is_fortran_contiguous() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let strides = s.col_major_strides();
        assert!(s.is_fortran_contiguous(&strides));
    }

    #[test]
    fn test_stride_info() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let info = s.stride_info();
        assert_eq!(info.shape(), &[2, 3, 4]);
        assert_eq!(info.strides(), &[12, 4, 1]);
        assert_eq!(info.numel(), 24);
        assert!(info.is_contiguous());
    }

    #[test]
    fn test_fortran_stride_info() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let info = s.fortran_stride_info();
        assert_eq!(info.strides(), &[1, 2, 6]);
        assert!(info.is_fortran_contiguous());
    }

    // =========================================================================
    // Free Function: row_major_strides / col_major_strides
    // =========================================================================

    #[test]
    fn test_row_major_strides_fn() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(row_major_strides(&[5]), vec![1]);
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_col_major_strides_fn() {
        assert_eq!(col_major_strides(&[2, 3, 4]), vec![1, 2, 6]);
    }

    // =========================================================================
    // Broadcasting Tests
    // =========================================================================

    #[test]
    fn test_broadcast_shapes_simple() {
        assert_eq!(broadcast_shapes(&[2, 1], &[1, 3]), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shapes_same() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shapes_different_rank() {
        assert_eq!(broadcast_shapes(&[3], &[2, 3]), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        assert_eq!(broadcast_shapes(&[], &[2, 3]), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[2, 3], &[]), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shapes_one_is_one() {
        assert_eq!(broadcast_shapes(&[1, 1], &[2, 3]), vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_shapes_incompatible() {
        broadcast_shapes(&[2, 3], &[4, 5]);
    }

    #[test]
    fn test_broadcast_shapes_multi() {
        assert_eq!(
            broadcast_shapes_multi(&[&[2, 1, 4], &[1, 3, 1], &[2, 3, 4]]),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_shapes_multi_empty() {
        assert_eq!(broadcast_shapes_multi(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_broadcast_shapes_multi_single() {
        assert_eq!(broadcast_shapes_multi(&[&[2, 3]]), vec![2, 3]);
    }

    #[test]
    fn test_can_broadcast() {
        assert!(can_broadcast(&[2, 1], &[1, 3]));
        assert!(can_broadcast(&[2, 3], &[2, 3]));
        assert!(can_broadcast(&[3], &[2, 3]));
    }

    #[test]
    fn test_cannot_broadcast() {
        assert!(!can_broadcast(&[2, 3], &[4, 5]));
        assert!(!can_broadcast(&[2, 3], &[2, 4]));
    }

    #[test]
    fn test_broadcast_method() {
        let result = Shape::broadcast_shapes(&[&Shape::from_dims(&[2, 1]), &Shape::from_dims(&[1, 3])]).unwrap().to_vec();
        assert_eq!(result, vec![2, 3]);
    }

    // =========================================================================
    // Conv Output Shape Tests
    // =========================================================================

    #[test]
    fn test_conv_output_shape_basic() {
        let out = compute_conv_output_shape(
            &[1, 3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            &[1, 1],
            1,
            16,
        );
        assert_eq!(out, vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_conv_output_shape_stride2() {
        let out = compute_conv_output_shape(
            &[1, 3, 32, 32],
            &[3, 3],
            &[2, 2],
            &[1, 1],
            &[1, 1],
            1,
            16,
        );
        assert_eq!(out, vec![1, 16, 16, 16]);
    }

    #[test]
    fn test_conv_output_shape_padding2() {
        let out = compute_conv_output_shape(
            &[1, 3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[2, 2],
            &[1, 1],
            1,
            16,
        );
        assert_eq!(out, vec![1, 16, 34, 34]);
    }

    #[test]
    fn test_conv_output_shape_dilation2() {
        let out = compute_conv_output_shape(
            &[1, 3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            &[2, 2],
            1,
            16,
        );
        assert_eq!(out, vec![1, 16, 30, 30]);
    }

    #[test]
    fn test_conv_output_shape_1x1() {
        let out = compute_conv_output_shape(
            &[1, 3, 32, 32],
            &[1, 1],
            &[1, 1],
            &[0, 0],
            &[1, 1],
            1,
            16,
        );
        assert_eq!(out, vec![1, 16, 32, 32]);
    }

    // =========================================================================
    // Pool Output Shape Tests
    // =========================================================================

    #[test]
    fn test_pool_output_shape_basic() {
        let out = compute_pool_output_shape(
            &[1, 3, 32, 32],
            &[2, 2],
            &[2, 2],
            &[0, 0],
        );
        assert_eq!(out, vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_pool_output_shape_padding() {
        let out = compute_pool_output_shape(
            &[1, 3, 32, 32],
            &[3, 3],
            &[2, 2],
            &[1, 1],
        );
        assert_eq!(out, vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_pool_output_shape_global() {
        let out = compute_pool_output_shape(
            &[1, 3, 32, 32],
            &[32, 32],
            &[1, 1],
            &[0, 0],
        );
        assert_eq!(out, vec![1, 3, 1, 1]);
    }

    // =========================================================================
    // Transpose Shape Tests
    // =========================================================================

    #[test]
    fn test_transpose_shape() {
        assert_eq!(compute_transpose_shape(&[2, 3, 4], &[2, 0, 1]), vec![4, 2, 3]);
    }

    #[test]
    fn test_transpose_shape_identity() {
        assert_eq!(compute_transpose_shape(&[2, 3, 4], &[0, 1, 2]), vec![2, 3, 4]);
    }

    #[test]
    fn test_transpose_shape_2d() {
        assert_eq!(compute_transpose_shape(&[2, 3], &[1, 0]), vec![3, 2]);
    }

    // =========================================================================
    // Reshape Shape Tests
    // =========================================================================

    #[test]
    fn test_reshape_shape() {
        assert_eq!(compute_reshape_shape(&[2, 3, 4], &[6, 4]), vec![6, 4]);
        assert_eq!(compute_reshape_shape(&[2, 3, 4], &[2, -1]), vec![2, 12]);
    }

    #[test]
    fn test_reshape_shape_infer() {
        assert_eq!(compute_reshape_shape(&[2, 3, 4], &[-1]), vec![24]);
        assert_eq!(compute_reshape_shape(&[2, 3, 4], &[-1, 4]), vec![6, 4]);
        assert_eq!(compute_reshape_shape(&[2, 3, 4], &[2, -1, 2]), vec![2, 6, 2]);
    }

    #[test]
    fn test_reshape_shape_same() {
        assert_eq!(compute_reshape_shape(&[2, 3], &[2, 3]), vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_reshape_shape_incompatible() {
        compute_reshape_shape(&[2, 3], &[5, 5]);
    }

    #[test]
    #[should_panic]
    fn test_reshape_shape_multiple_neg_one() {
        compute_reshape_shape(&[2, 3, 4], &[-1, -1, 2]);
    }

    // =========================================================================
    // ShapeIndex Tests
    // =========================================================================

    #[test]
    fn test_shape_index_new() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::Index(0),
            ShapeIndexType::All,
        ]);
        assert_eq!(idx.ndim(), 2);
    }

    #[test]
    fn test_shape_index_all() {
        let idx = ShapeIndex::all(3);
        assert_eq!(idx.ndim(), 3);
    }

    #[test]
    fn test_shape_index_output_shape_single() {
        let idx = ShapeIndex::new(vec![ShapeIndexType::Index(0)]);
        let input = Shape::from_dims(&[5, 3]);
        let output = idx.output_shape(&input);
        assert_eq!(output.as_slice(), &[3]);
    }

    #[test]
    fn test_shape_index_output_shape_all() {
        let idx = ShapeIndex::all(3);
        let input = Shape::from_dims(&[2, 3, 4]);
        let output = idx.output_shape(&input);
        assert_eq!(output.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_index_output_shape_slice() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::Slice { start: Some(1), end: Some(4), step: None },
        ]);
        let input = Shape::vector(10);
        let output = idx.output_shape(&input);
        assert_eq!(output.as_slice(), &[3]);
    }

    #[test]
    fn test_shape_index_output_shape_with_step() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::Slice { start: Some(0), end: Some(10), step: Some(2) },
        ]);
        let input = Shape::vector(10);
        let output = idx.output_shape(&input);
        assert_eq!(output.as_slice(), &[5]);
    }

    #[test]
    fn test_shape_index_output_shape_new_axis() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::Index(0),
            ShapeIndexType::NewAxis,
        ]);
        let input = Shape::from_dims(&[5, 3]);
        let output = idx.output_shape(&input);
        assert_eq!(output.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_shape_index_num_new_axes() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::NewAxis,
            ShapeIndexType::All,
            ShapeIndexType::NewAxis,
        ]);
        assert_eq!(idx.num_new_axes(), 2);
    }

    #[test]
    fn test_shape_index_consumed_dims() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::Index(0),
            ShapeIndexType::All,
            ShapeIndexType::NewAxis,
        ]);
        assert_eq!(idx.consumed_dims(), 2);
    }

    // =========================================================================
    // StrideInfo Tests
    // =========================================================================

    #[test]
    fn test_stride_info_new() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![12, 4, 1]);
        assert_eq!(info.shape(), &[2, 3, 4]);
        assert_eq!(info.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_stride_info_numel() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![12, 4, 1]);
        assert_eq!(info.numel(), 24);
    }

    #[test]
    fn test_stride_info_compute_index() {
        let info = StrideInfo::new(vec![2, 3], vec![3, 1]);
        assert_eq!(info.compute_index(&[0, 0]), 0);
        assert_eq!(info.compute_index(&[1, 2]), 5);
    }

    #[test]
    fn test_stride_info_is_contiguous() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![12, 4, 1]);
        assert!(info.is_contiguous());
    }

    #[test]
    fn test_stride_info_not_contiguous() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![4, 1, 12]);
        assert!(!info.is_contiguous());
    }

    #[test]
    fn test_stride_info_is_fortran_contiguous() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![1, 2, 6]);
        assert!(info.is_fortran_contiguous());
    }

    #[test]
    fn test_stride_info_stride_at() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![12, 4, 1]);
        assert_eq!(info.stride_at(0), 12);
        assert_eq!(info.stride_at(2), 1);
    }

    #[test]
    fn test_stride_info_size_at() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![12, 4, 1]);
        assert_eq!(info.size_at(0), 2);
        assert_eq!(info.size_at(2), 4);
    }

    #[test]
    fn test_stride_info_memory_range() {
        let info = StrideInfo::new(vec![2, 3], vec![3, 1]);
        let (min, max) = info.memory_range(4);
        assert_eq!(min, 0);
        assert_eq!(max, 24); // 6 elements * 4 bytes
    }

    #[test]
    fn test_stride_info_same_layout() {
        let a = StrideInfo::new(vec![2, 3], vec![3, 1]);
        let b = StrideInfo::new(vec![2, 3], vec![3, 1]);
        assert!(a.same_layout(&b));
    }

    #[test]
    fn test_stride_info_different_layout() {
        let a = StrideInfo::new(vec![2, 3], vec![3, 1]);
        let b = StrideInfo::new(vec![2, 3], vec![1, 2]);
        assert!(!a.same_layout(&b));
    }

    #[test]
    fn test_stride_info_display() {
        let info = StrideInfo::new(vec![2, 3], vec![3, 1]);
        let display = format!("{}", info);
        assert!(display.contains("Shape: [2, 3]"));
        assert!(display.contains("Strides: [3, 1]"));
    }

    #[test]
    fn test_stride_info_ndim() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![12, 4, 1]);
        assert_eq!(info.ndim(), 3);
    }

    // =========================================================================
    // Display Tests
    // =========================================================================

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Shape::from_dims(&[2, 3, 4])), "[2, 3, 4]");
        assert_eq!(format!("{}", Shape::scalar()), "[]");
        assert_eq!(format!("{}", Shape::vector(5)), "[5]");
    }

    // =========================================================================
    // Trait Implementation Tests
    // =========================================================================

    #[test]
    fn test_deref() {
        let s = Shape::from_dims(&[2, 3]);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], 2);
        assert_eq!(s[1], 3);
    }

    #[test]
    fn test_deref_mut() {
        let mut s = Shape::from_dims(&[2, 3]);
        s[0] = 5;
        assert_eq!(s[0], 5);
    }

    #[test]
    fn test_index_trait() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s[0], 2);
        assert_eq!(s[1], 3);
        assert_eq!(s[2], 4);
    }

    #[test]
    fn test_index_mut_trait() {
        let mut s = Shape::from_dims(&[2, 3, 4]);
        s[1] = 10;
        assert_eq!(s[1], 10);
    }

    #[test]
    fn test_default() {
        let s = Shape::default();
        assert!(s.is_scalar());
    }

    #[test]
    fn test_from_vec() {
        let s = Shape::from(vec![2, 3]);
        assert_eq!(s.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_into_vec() {
        let s = Shape::from_dims(&[2, 3]);
        let v: Vec<usize> = s.into();
        assert_eq!(v, vec![2, 3]);
    }

    #[test]
    fn test_from_slice() {
        let s = Shape::from(&[2usize, 3, 4][..]);
        assert_eq!(s.as_slice(), &[2, 3, 4]);
    }

    // =========================================================================
    // Equality and Hash Tests
    // =========================================================================

    #[test]
    fn test_equality() {
        let a = Shape::from_dims(&[2, 3]);
        let b = Shape::from_dims(&[2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_inequality() {
        let a = Shape::from_dims(&[2, 3]);
        let b = Shape::from_dims(&[3, 2]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Shape::from_dims(&[2, 3]));
        set.insert(Shape::from_dims(&[2, 3]));
        set.insert(Shape::from_dims(&[4, 5]));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_clone() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    #[test]
    fn test_debug() {
        let s = Shape::from_dims(&[2, 3]);
        let debug = format!("{:?}", s);
        assert!(debug.contains("2"));
        assert!(debug.contains("3"));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_large_shape() {
        let dims: Vec<usize> = (1..=100).collect();
        let s = Shape::new(dims);
        assert_eq!(s.ndim(), 100);
        assert_eq!(s.first_dim(), 1);
        assert_eq!(s.last_dim(), 100);
    }

    #[test]
    fn test_shape_size_one_all() {
        let s = Shape::cube(5, 1);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.squeeze().ndim(), 0);
    }

    #[test]
    fn test_broadcast_with_scalar() {
        assert_eq!(broadcast_shapes(&[], &[2, 3, 4]), vec![2, 3, 4]);
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[]), vec![2, 3, 4]);
    }

    #[test]
    fn test_conv_output_shape_batch() {
        let out = compute_conv_output_shape(
            &[8, 3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            &[1, 1],
            1,
            64,
        );
        assert_eq!(out[0], 8);
        assert_eq!(out[1], 64);
    }

    #[test]
    fn test_permute_round_trip() {
        let s = Shape::from_dims(&[2, 3, 4, 5]);
        let perm = vec![2, 0, 3, 1];
        let permuted = s.permuted(&perm);
        let inv_perm = s.inverse_perm(&perm);
        let restored = permuted.permuted(&inv_perm);
        assert_eq!(s, restored);
    }

    #[test]
    fn test_reshape_preserve_numel() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let reshaped = compute_reshape_shape(s.as_slice(), &[6, -1]);
        let new_shape = Shape::from_dims(&reshaped);
        assert_eq!(new_shape.numel(), s.numel());
    }

    #[test]
    fn test_iter_mut() {
        let mut s = Shape::from_dims(&[1, 2, 3]);
        for dim in s.iter_mut() {
            *dim *= 2;
        }
        assert_eq!(s.as_slice(), &[2, 4, 6]);
    }

    #[test]
    fn test_shape_index_output_shape_complex() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::Slice { start: None, end: Some(3), step: None },
            ShapeIndexType::Index(1),
            ShapeIndexType::NewAxis,
            ShapeIndexType::All,
        ]);
        let input = Shape::from_dims(&[10, 20, 30]);
        let output = idx.output_shape(&input);
        assert_eq!(output.ndim(), 3); // Index removes dim, NewAxis adds dim
        assert_eq!(output.size_at(0), 3);
        assert_eq!(output.size_at(1), 1); // NewAxis
        assert_eq!(output.size_at(2), 30);
    }

    // =========================================================================
    // Additional Constructor Tests
    // =========================================================================

    #[test]
    fn test_from_vec_macro() {
        let s = Shape::from(vec![2, 3, 4]);
        assert_eq!(s.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_from_empty_vec() {
        let s = Shape::from(vec![]);
        assert!(s.is_scalar());
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn test_from_slice_ref() {
        let dims = [2, 3, 4];
        let s = Shape::from(dims.as_slice());
        assert_eq!(s.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_batched_multiple_batch_dims() {
        let s = Shape::batched(&[4, 2], &[3, 224, 224]);
        assert_eq!(s.ndim(), 5);
        assert_eq!(s.as_slice(), &[4, 2, 3, 224, 224]);
    }

    #[test]
    fn test_batched_no_batch() {
        let s = Shape::batched(&[], &[3, 224, 224]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.as_slice(), &[3, 224, 224]);
    }

    // =========================================================================
    // Additional Broadcasting Tests
    // =========================================================================

    #[test]
    fn test_broadcast_three_shapes() {
        let r = broadcast_shapes_multi(&[&[2, 1, 4], &[1, 3, 1], &[2, 3, 4]]);
        assert_eq!(r, vec![2, 3, 4]);
    }

    #[test]
    fn test_broadcast_scalar_with_3d() {
        let r = broadcast_shapes(&[], &[2, 3, 4]);
        assert_eq!(r, vec![2, 3, 4]);
    }

    #[test]
    fn test_broadcast_vector_with_matrix() {
        let r = broadcast_shapes(&[4], &[3, 4]);
        assert_eq!(r, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_matrix_with_vector() {
        let r = broadcast_shapes(&[3, 4], &[4]);
        assert_eq!(r, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_ones_with_ones() {
        let r = broadcast_shapes(&[1, 1, 1], &[1, 1, 1]);
        assert_eq!(r, vec![1, 1, 1]);
    }

    #[test]
    fn test_can_broadcast_different_ranks() {
        assert!(can_broadcast(&[5], &[2, 3, 5]));
        assert!(can_broadcast(&[1, 3], &[2, 1, 3]));
        assert!(can_broadcast(&[], &[2, 3, 4]));
    }

    #[test]
    fn test_can_not_broadcast_incompatible() {
        assert!(!can_broadcast(&[2], &[3]));
        assert!(!can_broadcast(&[2, 3], &[2, 4]));
        assert!(!can_broadcast(&[2, 3, 5], &[2, 4, 5]));
    }

    #[test]
    fn test_can_broadcast_trivial() {
        assert!(can_broadcast(&[], &[]));
        assert!(can_broadcast(&[5], &[5]));
    }

    // =========================================================================
    // Additional Conv Tests
    // =========================================================================

    #[test]
    fn test_conv_same_padding() {
        let out = compute_conv_output_shape(
            &[1, 3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            &[1, 1],
            1,
            64,
        );
        assert_eq!(out, vec![1, 64, 32, 32]);
    }

    #[test]
    fn test_conv_large_kernel() {
        let out = compute_conv_output_shape(
            &[1, 3, 224, 224],
            &[7, 7],
            &[2, 2],
            &[3, 3],
            &[1, 1],
            1,
            64,
        );
        assert_eq!(out, vec![1, 64, 112, 112]);
    }

    #[test]
    fn test_conv_batch() {
        let out = compute_conv_output_shape(
            &[16, 3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            &[1, 1],
            1,
            64,
        );
        assert_eq!(out[0], 16);
    }

    #[test]
    fn test_conv_groups() {
        // With groups, out_channels per group determines output
        let out = compute_conv_output_shape(
            &[1, 4, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            &[1, 1],
            2,
            8,
        );
        assert_eq!(out[1], 8);
        assert_eq!(out[2], 32);
    }

    // =========================================================================
    // Additional Pool Tests
    // =========================================================================

    #[test]
    fn test_pool_same_padding() {
        let out = compute_pool_output_shape(
            &[1, 3, 32, 32],
            &[3, 3],
            &[1, 1],
            &[1, 1],
        );
        assert_eq!(out, vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_pool_non_square_kernel() {
        let out = compute_pool_output_shape(
            &[1, 3, 32, 64],
            &[2, 4],
            &[2, 4],
            &[0, 0],
        );
        assert_eq!(out, vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_pool_1x1() {
        let out = compute_pool_output_shape(
            &[1, 3, 32, 32],
            &[1, 1],
            &[1, 1],
            &[0, 0],
        );
        assert_eq!(out, vec![1, 3, 32, 32]);
    }

    // =========================================================================
    // Additional Transpose Tests
    // =========================================================================

    #[test]
    fn test_transpose_3d() {
        assert_eq!(compute_transpose_shape(&[2, 3, 4], &[1, 0, 2]), vec![3, 2, 4]);
        assert_eq!(compute_transpose_shape(&[2, 3, 4], &[0, 2, 1]), vec![2, 4, 3]);
        assert_eq!(compute_transpose_shape(&[2, 3, 4], &[2, 1, 0]), vec![4, 3, 2]);
    }

    #[test]
    fn test_transpose_1d() {
        assert_eq!(compute_transpose_shape(&[5], &[0]), vec![5]);
    }

    // =========================================================================
    // Additional Reshape Tests
    // =========================================================================

    #[test]
    fn test_reshape_flatten() {
        assert_eq!(compute_reshape_shape(&[2, 3, 4], &[24]), vec![24]);
        assert_eq!(compute_reshape_shape(&[2, 3, 4], &[24, 1]), vec![24, 1]);
    }

    #[test]
    fn test_reshape_add_dims() {
        assert_eq!(compute_reshape_shape(&[24], &[2, 3, 4]), vec![2, 3, 4]);
        assert_eq!(compute_reshape_shape(&[24], &[1, 24]), vec![1, 24]);
    }

    #[test]
    #[should_panic(expected = "negative")]
    fn test_reshape_negative_not_minus_one() {
        compute_reshape_shape(&[6], &[2, -2]);
    }

    // =========================================================================
    // Shape Permutation Round Trip Tests
    // =========================================================================

    #[test]
    fn test_permute_round_trip_4d() {
        let s = Shape::from_dims(&[2, 3, 4, 5]);
        let perm = vec![3, 1, 0, 2];
        let permuted = s.permuted(&perm);
        let inv = s.inverse_perm(&perm);
        let restored = permuted.permuted(&inv);
        assert_eq!(s, restored);
    }

    #[test]
    fn test_inverse_perm_is_inverse() {
        let perms = vec![
            vec![0, 1, 2],
            vec![2, 1, 0],
            vec![1, 2, 0],
            vec![2, 0, 1],
            vec![0, 2, 1],
        ];
        for perm in &perms {
            let inv = Shape::from_dims(&[1; 3]).inverse_perm(perm);
            // Apply perm then inv should give identity
            let double_inv: Vec<usize> = inv.iter().map(|&i| perm[i]).collect();
            assert_eq!(double_inv, vec![0, 1, 2]);
        }
    }

    // =========================================================================
    // Additional Stride Tests
    // =========================================================================

    #[test]
    fn test_stride_info_display_empty() {
        let info = StrideInfo::new(vec![], vec![]);
        let display = format!("{}", info);
        assert!(display.contains("Shape: []"));
    }

    #[test]
    fn test_stride_info_empty() {
        let info = StrideInfo::new(vec![], vec![]);
        assert_eq!(info.ndim(), 0);
        assert_eq!(info.numel(), 1);
        assert!(info.is_contiguous());
    }

    #[test]
    fn test_stride_info_single_dim() {
        let info = StrideInfo::new(vec![5], vec![1]);
        assert!(info.is_contiguous());
        assert_eq!(info.compute_index(&[3]), 3);
    }

    #[test]
    fn test_stride_info_not_fortran() {
        let info = StrideInfo::new(vec![2, 3, 4], vec![12, 4, 1]);
        assert!(!info.is_fortran_contiguous());
    }

    #[test]
    fn test_stride_info_both_contiguous_scalar() {
        let info = StrideInfo::new(vec![], vec![]);
        assert!(info.is_contiguous());
        assert!(info.is_fortran_contiguous());
    }

    // =========================================================================
    // ShapeIndex Extended Tests
    // =========================================================================

    #[test]
    fn test_shape_index_slice_full() {
        let idx = ShapeIndex::new(vec![ShapeIndexType::Slice {
            start: None, end: None, step: None,
        }]);
        let input = Shape::vector(10);
        let output = idx.output_shape(&input);
        assert_eq!(output.size_at(0), 10);
    }

    #[test]
    fn test_shape_index_slice_with_step() {
        let idx = ShapeIndex::new(vec![ShapeIndexType::Slice {
            start: Some(0), end: Some(10), step: Some(3),
        }]);
        let input = Shape::vector(10);
        let output = idx.output_shape(&input);
        assert_eq!(output.size_at(0), 4); // 0,3,6,9
    }

    #[test]
    fn test_shape_index_slice_empty_result() {
        let idx = ShapeIndex::new(vec![ShapeIndexType::Slice {
            start: Some(5), end: Some(3), step: None,
        }]);
        let input = Shape::vector(10);
        let output = idx.output_shape(&input);
        assert_eq!(output.size_at(0), 0);
    }

    #[test]
    fn test_shape_index_multiple_new_axis() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::NewAxis,
            ShapeIndexType::NewAxis,
            ShapeIndexType::All,
        ]);
        let input = Shape::vector(5);
        let output = idx.output_shape(&input);
        assert_eq!(output.ndim(), 3);
        assert_eq!(output.size_at(0), 1);
        assert_eq!(output.size_at(1), 1);
        assert_eq!(output.size_at(2), 5);
    }

    #[test]
    fn test_shape_index_only_new_axes() {
        let idx = ShapeIndex::new(vec![
            ShapeIndexType::NewAxis,
            ShapeIndexType::NewAxis,
        ]);
        let input = Shape::scalar();
        let output = idx.output_shape(&input);
        assert_eq!(output.ndim(), 2);
        assert_eq!(output.as_slice(), &[1, 1]);
    }

    // =========================================================================
    // Shape Manipulation Extended Tests
    // =========================================================================

    #[test]
    fn test_squeeze_preserves_non_one_dims() {
        let s = Shape::from_dims(&[2, 3, 4, 5]);
        let squeezed = s.squeeze();
        assert_eq!(squeezed.as_slice(), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_squeeze_shape_no_match() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let squeezed = s.squeeze_shape(&[0, 1, 2]);
        assert_eq!(squeezed.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_pad_to_rank_no_change() {
        let s = Shape::from_dims(&[2, 3, 4, 5]);
        assert_eq!(s.pad_to_rank(3).as_slice(), &[2, 3, 4, 5]);
        assert_eq!(s.pad_to_rank(4).as_slice(), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_trim_trailing_ones_all_ones() {
        let s = Shape::from_dims(&[1, 1, 1]);
        assert_eq!(s.trim_trailing_ones().ndim(), 0);
    }

    #[test]
    fn test_trim_trailing_ones_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.trim_trailing_ones().ndim(), 0);
    }

    #[test]
    fn test_reversed_1d() {
        let s = Shape::vector(5);
        assert_eq!(s.reversed().as_slice(), &[5]);
    }

    #[test]
    fn test_reversed_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.reversed().ndim(), 0);
    }

    #[test]
    fn test_permuted_1d() {
        let s = Shape::vector(5);
        assert_eq!(s.permuted(&[0]).as_slice(), &[5]);
    }

    // =========================================================================
    // Compute Index Extended Tests
    // =========================================================================

    #[test]
    fn test_compute_index_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.compute_index(&[]), 0);
    }

    #[test]
    fn test_compute_index_1d() {
        let s = Shape::vector(5);
        assert_eq!(s.compute_index(&[0]), 0);
        assert_eq!(s.compute_index(&[4]), 4);
    }

    #[test]
    fn test_compute_index_4d() {
        let s = Shape::from_dims(&[2, 3, 4, 5]);
        assert_eq!(s.compute_index(&[0, 0, 0, 0]), 0);
        assert_eq!(s.compute_index(&[0, 0, 0, 1]), 1);
        assert_eq!(s.compute_index(&[0, 0, 1, 0]), 5);
        assert_eq!(s.compute_index(&[0, 1, 0, 0]), 20);
        assert_eq!(s.compute_index(&[1, 0, 0, 0]), 60);
        assert_eq!(s.compute_index(&[1, 2, 3, 4]), 119);
    }

    // =========================================================================
    // From Flat Index Extended Tests
    // =========================================================================

    #[test]
    fn test_from_flat_index_scalar() {
        let shape = vec![];
        assert_eq!(Shape::from_flat_index(0, &shape), Vec::<usize>::new());
    }

    #[test]
    fn test_from_flat_index_1d() {
        let shape = vec![5];
        assert_eq!(Shape::from_flat_index(3, &shape), vec![3]);
    }

    #[test]
    fn test_from_flat_index_last_element() {
        let shape = vec![2, 3, 4];
        assert_eq!(Shape::from_flat_index(23, &shape), vec![1, 2, 3]);
    }

    // =========================================================================
    // Row Major Strides Extended Tests
    // =========================================================================

    #[test]
    fn test_row_major_strides_empty() {
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_row_major_strides_with_zero_dim() {
        // Handle zero-size dimensions gracefully
        let s = Shape::from_dims(&[2, 0, 4]);
        let strides = s.row_major_strides();
        // Last dim should be 1 even if size is 0
        assert_eq!(strides[2], 1);
    }

    #[test]
    fn test_col_major_strides_with_zero_dim() {
        let s = Shape::from_dims(&[0, 3]);
        let strides = s.col_major_strides();
        assert_eq!(strides[0], 1);
    }

    // =========================================================================
    // Memory Range Tests
    // =========================================================================

    #[test]
    fn test_stride_info_memory_range_empty() {
        let info = StrideInfo::new(vec![], vec![]);
        let (min, max) = info.memory_range(4);
        assert_eq!(min, 0);
        assert_eq!(max, 4);
    }

    #[test]
    fn test_stride_info_memory_range_1d() {
        let info = StrideInfo::new(vec![10], vec![1]);
        let (min, max) = info.memory_range(4);
        assert_eq!(min, 0);
        assert_eq!(max, 40);
    }

    #[test]
    fn test_stride_info_memory_range_different_sizes() {
        let info = StrideInfo::new(vec![3, 4], vec![4, 1]);
        let (min4, max4) = info.memory_range(4);
        assert_eq!(min4, 0);
        assert_eq!(max4, 48); // 12 elements * 4 bytes
        let (min8, max8) = info.memory_range(8);
        assert_eq!(max8, 96); // 12 elements * 8 bytes
    }

    // =========================================================================
    // Broadcast Shapes Multi Edge Tests
    // =========================================================================

    #[test]
    fn test_broadcast_shapes_multi_single_edge() {
        assert_eq!(broadcast_shapes_multi(&[&[5]]), vec![5]);
    }

    #[test]
    fn test_broadcast_shapes_multi_four() {
        let r = broadcast_shapes_multi(&[&[1], &[1, 1], &[1, 3, 1], &[1, 3, 4]]);
        assert_eq!(r, vec![1, 3, 4]);
    }

    // =========================================================================
    // Shape Utility Tests
    // =========================================================================

    #[test]
    fn test_product_except_all_dims() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert_eq!(s.product_except(0), 12);
        assert_eq!(s.product_except(1), 8);
        assert_eq!(s.product_except(2), 6);
    }

    #[test]
    fn test_product_except_1d() {
        let s = Shape::vector(10);
        assert_eq!(s.product_except(0), 1);
    }

    #[test]
    fn test_product_except_matrix() {
        let s = Shape::matrix(5, 7);
        assert_eq!(s.product_except(0), 7);
        assert_eq!(s.product_except(1), 5);
    }

    // =========================================================================
    // Additional Conv Edge Cases
    // =========================================================================

    #[test]
    fn test_conv_output_shape_5x5_kernel() {
        let out = compute_conv_output_shape(
            &[1, 3, 28, 28],
            &[5, 5],
            &[1, 1],
            &[2, 2],
            &[1, 1],
            1,
            32,
        );
        assert_eq!(out, vec![1, 32, 28, 28]);
    }

    #[test]
    fn test_conv_output_shape_stride_3() {
        let out = compute_conv_output_shape(
            &[1, 3, 64, 64],
            &[3, 3],
            &[3, 3],
            &[1, 1],
            &[1, 1],
            1,
            16,
        );
        assert_eq!(out, vec![1, 16, 22, 22]);
    }

    // =========================================================================
    // Broadcasting Edge Cases
    // =========================================================================

    #[test]
    fn test_broadcast_shape_method_5d() {
        let s = Shape::from_dims(&[1, 1, 1, 1, 1]);
        let result = s.broadcast_shape(&[2, 3, 4, 5, 6]);
        assert_eq!(result.as_slice(), &[2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_method_same() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let result = s.broadcast_shape(&[2, 3, 4]);
        assert_eq!(result.as_slice(), &[2, 3, 4]);
    }

    // =========================================================================
    // Shape Debug Tests
    // =========================================================================

    #[test]
    fn test_debug_format() {
        let s = Shape::from_dims(&[2, 3]);
        let debug = format!("{:?}", s);
        assert!(debug.contains("2"));
        assert!(debug.contains("3"));
    }

    #[test]
    fn test_debug_scalar() {
        let s = Shape::scalar();
        let debug = format!("{:?}", s);
        assert!(debug.is_empty() || debug.contains("Shape"));
    }

    // =========================================================================
    // From Flat Index Round Trip
    // =========================================================================

    #[test]
    fn test_from_flat_index_round_trip_2d() {
        let shape = vec![3, 4];
        for flat in 0..12 {
            let indices = Shape::from_flat_index(flat, &shape);
            let s = Shape::from_dims(&shape);
            assert_eq!(s.compute_index(&indices), flat);
        }
    }

    #[test]
    fn test_from_flat_index_round_trip_3d() {
        let shape = vec![2, 3, 4];
        for flat in 0..24 {
            let indices = Shape::from_flat_index(flat, &shape);
            let s = Shape::from_dims(&shape);
            assert_eq!(s.compute_index(&indices), flat);
        }
    }

    #[test]
    fn test_shape_merge() {
        let s1 = Shape::from_dims(&[2, 3]);
        let s2 = Shape::from_dims(&[4, 5, 6]);
        let m = Shape::merge(&s1, &s2);
        assert_eq!(m.as_slice(), &[2, 3, 4, 5, 6]);
        assert_eq!(m.numel(), 720);
    }

    #[test]
    fn test_shape_split() {
        let s = Shape::from_dims(&[1, 2, 3, 4, 5]);
        let (left, right) = s.split(2);
        assert_eq!(left.as_slice(), &[1, 2]);
        assert_eq!(right.as_slice(), &[3, 4, 5]);
    }

    #[test]
    fn test_shape_transposed() {
        let s = Shape::from_dims(&[2, 3, 4]);
        let t = s.transposed(0, 2);
        assert_eq!(t.as_slice(), &[4, 3, 2]);
    }

    #[test]
    fn test_shape_is_broadcastable_with() {
        let s1 = Shape::from_dims(&[2, 1, 4]);
        let s2 = Shape::from_dims(&[3, 4]);
        let s3 = Shape::from_dims(&[2, 3, 5]);
        assert!(s1.is_broadcastable_with(&s2));
        assert!(!s1.is_broadcastable_with(&s3));
    }

    #[test]
    fn test_shape_broadcast_to_success() {
        let s = Shape::from_dims(&[1, 4]);
        let b = s.broadcast_to(&[2, 4]).unwrap();
        assert_eq!(b.as_slice(), &[2, 4]);
    }

    #[test]
    fn test_shape_broadcast_to_error() {
        let s = Shape::from_dims(&[3, 4]);
        assert!(s.broadcast_to(&[2, 4]).is_err());
    }

    #[test]
    fn test_shape_broadcast_shapes() {
        let s1 = Shape::from_dims(&[2, 1]);
        let s2 = Shape::from_dims(&[1, 3]);
        let s3 = Shape::from_dims(&[2, 3]);
        let common = Shape::broadcast_shapes(&[&s1, &s2, &s3]).unwrap();
        assert_eq!(common.as_slice(), &[2, 3]);
    }

    #[test]
    fn test_shape_narrow() {
        let s = Shape::from_dims(&[10, 20]);
        let n = s.narrow(0, 2, 5).unwrap();
        assert_eq!(n.as_slice(), &[5, 20]);
        assert!(s.narrow(0, 8, 5).is_err());
    }

    #[test]
    fn test_shape_expanded() {
        let s = Shape::from_dims(&[1, 3, 1]);
        let exp = s.expanded(&[4, 3, 5]).unwrap();
        assert_eq!(exp.as_slice(), &[4, 3, 5]);
        assert!(s.expanded(&[4, 2, 5]).is_err());
    }

    #[test]
    fn test_shape_validation() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert!(s.validate().is_ok());
        assert!(s.validate_index(&[1, 2, 3]).is_ok());
        assert!(s.validate_index(&[2, 2, 3]).is_err());
        assert!(s.validate_strides(&[12, 4, 1]).is_ok());
        assert!(s.validate_strides(&[12, 4]).is_err());
        assert!(s.validate_permutation(&[2, 0, 1]).is_ok());
        assert!(s.validate_permutation(&[2, 0, 2]).is_err());
    }
}
