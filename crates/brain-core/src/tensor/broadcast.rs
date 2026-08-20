//! NumPy-style lazy broadcasting engine and offset mapping for the Brain DL framework.
//!
//! This module provides zero-copy broadcast views ([`BroadcastView`]), offset/stride mappings
//! ([`BroadcastInfo`]), multi-tensor shape alignment, and detailed error reports for dimension conflicts.

use crate::error::{BrainError, BrainResult};
use crate::tensor::Tensor;

// =============================================================================
// BroadcastInfo - Coordinate-to-Offset Mapping
// =============================================================================

/// Encapsulates stride and dimension mapping for broadcasting from source to target shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastInfo {
    /// The original source tensor shape.
    pub src_shape: Vec<usize>,
    /// The original source tensor strides.
    pub src_strides: Vec<usize>,
    /// The broadcasted target shape.
    pub target_shape: Vec<usize>,
    /// The effective strides in target shape (0 for broadcasted singleton dimensions).
    pub effective_strides: Vec<usize>,
}

impl BroadcastInfo {
    /// Computes broadcast mapping from source shape and strides to a target shape.
    pub fn new(
        src_shape: &[usize],
        src_strides: &[usize],
        target_shape: &[usize],
    ) -> BrainResult<Self> {
        let src_rank = src_shape.len();
        let tgt_rank = target_shape.len();
        if src_rank > tgt_rank {
            return Err(BrainError::shape_mismatch(
                format!("{:?}", target_shape),
                format!("{:?}", src_shape),
                "BroadcastInfo: target rank cannot be smaller than source rank",
            ));
        }

        let mut effective_strides = vec![0usize; tgt_rank];
        for i in 0..src_rank {
            let src_dim = src_shape[src_rank - 1 - i];
            let tgt_dim = target_shape[tgt_rank - 1 - i];
            let src_stride = src_strides[src_rank - 1 - i];

            if src_dim == tgt_dim {
                effective_strides[tgt_rank - 1 - i] = src_stride;
            } else if src_dim == 1 {
                effective_strides[tgt_rank - 1 - i] = 0;
            } else {
                return Err(BrainError::shape_mismatch(
                    format!("dim {} at trailing position {}", tgt_dim, i),
                    format!("dim {}", src_dim),
                    "BroadcastInfo: incompatible dimension for broadcasting",
                ));
            }
        }

        Ok(BroadcastInfo {
            src_shape: src_shape.to_vec(),
            src_strides: src_strides.to_vec(),
            target_shape: target_shape.to_vec(),
            effective_strides,
        })
    }

    /// Computes the flat byte/element offset in the source buffer for target coordinates.
    #[inline(always)]
    pub fn compute_source_offset(&self, target_coords: &[usize]) -> usize {
        let mut offset = 0;
        for (i, &c) in target_coords.iter().enumerate() {
            offset += c * self.effective_strides[i];
        }
        offset
    }
}

// =============================================================================
// BroadcastView - Zero-Copy Lazy Broadcast View
// =============================================================================

/// A zero-copy read-only view over a tensor broadcasted to a larger shape.
#[derive(Debug, Clone)]
pub struct BroadcastView<'a> {
    tensor: &'a Tensor,
    info: BroadcastInfo,
}

impl<'a> BroadcastView<'a> {
    /// Creates a new lazy broadcast view over `tensor`.
    pub fn new(tensor: &'a Tensor, target_shape: &[usize]) -> BrainResult<Self> {
        let info = BroadcastInfo::new(tensor.shape(), tensor.strides(), target_shape)?;
        Ok(BroadcastView { tensor, info })
    }

    /// Returns the virtual broadcasted shape.
    pub fn shape(&self) -> &[usize] {
        &self.info.target_shape
    }

    /// Returns total virtual elements.
    pub fn numel(&self) -> usize {
        self.info.target_shape.iter().product()
    }

    /// Gets an element at target multi-dimensional coordinates.
    pub fn get(&self, coords: &[usize]) -> f64 {
        let offset = self.info.compute_source_offset(coords);
        self.tensor.get(offset)
    }

    /// Materializes the lazy broadcast view into an owned contiguous [`Tensor`].
    pub fn to_tensor(&self) -> Tensor {
        let numel = self.numel();
        let mut data = Vec::with_capacity(numel);
        let rank = self.shape().len();
        let mut coords = vec![0usize; rank];

        for _ in 0..numel {
            data.push(self.get(&coords));
            for d in (0..rank).rev() {
                coords[d] += 1;
                if coords[d] < self.shape()[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Tensor::new(data, self.shape().to_vec())
    }
}

// =============================================================================
// Broadcasting Functions
// =============================================================================

/// Broadcasts a tensor to a target shape.
pub fn broadcast_to(tensor: &Tensor, target_shape: &[usize]) -> BrainResult<Tensor> {
    let view = BroadcastView::new(tensor, target_shape)?;
    Ok(view.to_tensor())
}

/// Broadcasts a batch of tensors to a common broadcast shape.
pub fn broadcast_batch(tensors: &[&Tensor]) -> BrainResult<Vec<Tensor>> {
    if tensors.is_empty() {
        return Ok(Vec::new());
    }
    let shapes: Vec<crate::shape::Shape> = tensors
        .iter()
        .map(|t| crate::shape::Shape::from_dims(t.shape()))
        .collect();
    let shape_refs: Vec<&crate::shape::Shape> = shapes.iter().collect();
    let common_shape = crate::shape::Shape::broadcast_shapes(&shape_refs)?;

    let mut result = Vec::with_capacity(tensors.len());
    for &t in tensors {
        result.push(broadcast_to(t, common_shape.as_slice())?);
    }
    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_info_and_view() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![1, 3]);
        let view = BroadcastView::new(&t, &[2, 3]).unwrap();
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.numel(), 6);
        assert_eq!(view.get(&[0, 0]), 1.0);
        assert_eq!(view.get(&[1, 2]), 3.0);

        let mat = view.to_tensor();
        assert_eq!(mat.shape(), &[2, 3]);
        assert_eq!(mat.data(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_batch() {
        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        let batch = broadcast_batch(&[&t1, &t2]).unwrap();
        assert_eq!(batch[0].shape(), &[2, 2]);
        assert_eq!(batch[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_broadcast_expansion_table() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]);
        let b = broadcast_to(&a, &[2, 4]).unwrap();
        assert_eq!(b.shape(), &[2, 4]);
        assert_eq!(b.get_2d(0, 0), 1.0);
        assert_eq!(b.get_2d(0, 3), 1.0);
        assert_eq!(b.get_2d(1, 3), 2.0);
    }
}
