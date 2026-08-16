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
    pub fn new(src_shape: &[usize], src_strides: &[usize], target_shape: &[usize]) -> BrainResult<Self> {
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
    fn test_broadcast_stress_case_001() {
        let t = Tensor::full(vec![1, 2], 1.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 1.0);
        assert_eq!(b.get(5), 1.0);
    }

    #[test]
    fn test_broadcast_stress_case_002() {
        let t = Tensor::full(vec![1, 2], 2.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 2.0);
        assert_eq!(b.get(5), 2.0);
    }

    #[test]
    fn test_broadcast_stress_case_003() {
        let t = Tensor::full(vec![1, 2], 3.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 3.0);
        assert_eq!(b.get(5), 3.0);
    }

    #[test]
    fn test_broadcast_stress_case_004() {
        let t = Tensor::full(vec![1, 2], 4.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 4.0);
        assert_eq!(b.get(5), 4.0);
    }

    #[test]
    fn test_broadcast_stress_case_005() {
        let t = Tensor::full(vec![1, 2], 5.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 5.0);
        assert_eq!(b.get(5), 5.0);
    }

    #[test]
    fn test_broadcast_stress_case_006() {
        let t = Tensor::full(vec![1, 2], 6.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 6.0);
        assert_eq!(b.get(5), 6.0);
    }

    #[test]
    fn test_broadcast_stress_case_007() {
        let t = Tensor::full(vec![1, 2], 7.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 7.0);
        assert_eq!(b.get(5), 7.0);
    }

    #[test]
    fn test_broadcast_stress_case_008() {
        let t = Tensor::full(vec![1, 2], 8.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 8.0);
        assert_eq!(b.get(5), 8.0);
    }

    #[test]
    fn test_broadcast_stress_case_009() {
        let t = Tensor::full(vec![1, 2], 9.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 9.0);
        assert_eq!(b.get(5), 9.0);
    }

    #[test]
    fn test_broadcast_stress_case_010() {
        let t = Tensor::full(vec![1, 2], 10.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 10.0);
        assert_eq!(b.get(5), 10.0);
    }

    #[test]
    fn test_broadcast_stress_case_011() {
        let t = Tensor::full(vec![1, 2], 11.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 11.0);
        assert_eq!(b.get(5), 11.0);
    }

    #[test]
    fn test_broadcast_stress_case_012() {
        let t = Tensor::full(vec![1, 2], 12.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 12.0);
        assert_eq!(b.get(5), 12.0);
    }

    #[test]
    fn test_broadcast_stress_case_013() {
        let t = Tensor::full(vec![1, 2], 13.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 13.0);
        assert_eq!(b.get(5), 13.0);
    }

    #[test]
    fn test_broadcast_stress_case_014() {
        let t = Tensor::full(vec![1, 2], 14.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 14.0);
        assert_eq!(b.get(5), 14.0);
    }

    #[test]
    fn test_broadcast_stress_case_015() {
        let t = Tensor::full(vec![1, 2], 15.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 15.0);
        assert_eq!(b.get(5), 15.0);
    }

    #[test]
    fn test_broadcast_stress_case_016() {
        let t = Tensor::full(vec![1, 2], 16.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 16.0);
        assert_eq!(b.get(5), 16.0);
    }

    #[test]
    fn test_broadcast_stress_case_017() {
        let t = Tensor::full(vec![1, 2], 17.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 17.0);
        assert_eq!(b.get(5), 17.0);
    }

    #[test]
    fn test_broadcast_stress_case_018() {
        let t = Tensor::full(vec![1, 2], 18.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 18.0);
        assert_eq!(b.get(5), 18.0);
    }

    #[test]
    fn test_broadcast_stress_case_019() {
        let t = Tensor::full(vec![1, 2], 19.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 19.0);
        assert_eq!(b.get(5), 19.0);
    }

    #[test]
    fn test_broadcast_stress_case_020() {
        let t = Tensor::full(vec![1, 2], 20.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 20.0);
        assert_eq!(b.get(5), 20.0);
    }

    #[test]
    fn test_broadcast_stress_case_021() {
        let t = Tensor::full(vec![1, 2], 21.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 21.0);
        assert_eq!(b.get(5), 21.0);
    }

    #[test]
    fn test_broadcast_stress_case_022() {
        let t = Tensor::full(vec![1, 2], 22.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 22.0);
        assert_eq!(b.get(5), 22.0);
    }

    #[test]
    fn test_broadcast_stress_case_023() {
        let t = Tensor::full(vec![1, 2], 23.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 23.0);
        assert_eq!(b.get(5), 23.0);
    }

    #[test]
    fn test_broadcast_stress_case_024() {
        let t = Tensor::full(vec![1, 2], 24.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 24.0);
        assert_eq!(b.get(5), 24.0);
    }

    #[test]
    fn test_broadcast_stress_case_025() {
        let t = Tensor::full(vec![1, 2], 25.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 25.0);
        assert_eq!(b.get(5), 25.0);
    }

    #[test]
    fn test_broadcast_stress_case_026() {
        let t = Tensor::full(vec![1, 2], 26.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 26.0);
        assert_eq!(b.get(5), 26.0);
    }

    #[test]
    fn test_broadcast_stress_case_027() {
        let t = Tensor::full(vec![1, 2], 27.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 27.0);
        assert_eq!(b.get(5), 27.0);
    }

    #[test]
    fn test_broadcast_stress_case_028() {
        let t = Tensor::full(vec![1, 2], 28.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 28.0);
        assert_eq!(b.get(5), 28.0);
    }

    #[test]
    fn test_broadcast_stress_case_029() {
        let t = Tensor::full(vec![1, 2], 29.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 29.0);
        assert_eq!(b.get(5), 29.0);
    }

    #[test]
    fn test_broadcast_stress_case_030() {
        let t = Tensor::full(vec![1, 2], 30.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 30.0);
        assert_eq!(b.get(5), 30.0);
    }

    #[test]
    fn test_broadcast_stress_case_031() {
        let t = Tensor::full(vec![1, 2], 31.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 31.0);
        assert_eq!(b.get(5), 31.0);
    }

    #[test]
    fn test_broadcast_stress_case_032() {
        let t = Tensor::full(vec![1, 2], 32.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 32.0);
        assert_eq!(b.get(5), 32.0);
    }

    #[test]
    fn test_broadcast_stress_case_033() {
        let t = Tensor::full(vec![1, 2], 33.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 33.0);
        assert_eq!(b.get(5), 33.0);
    }

    #[test]
    fn test_broadcast_stress_case_034() {
        let t = Tensor::full(vec![1, 2], 34.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 34.0);
        assert_eq!(b.get(5), 34.0);
    }

    #[test]
    fn test_broadcast_stress_case_035() {
        let t = Tensor::full(vec![1, 2], 35.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 35.0);
        assert_eq!(b.get(5), 35.0);
    }

    #[test]
    fn test_broadcast_stress_case_036() {
        let t = Tensor::full(vec![1, 2], 36.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 36.0);
        assert_eq!(b.get(5), 36.0);
    }

    #[test]
    fn test_broadcast_stress_case_037() {
        let t = Tensor::full(vec![1, 2], 37.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 37.0);
        assert_eq!(b.get(5), 37.0);
    }

    #[test]
    fn test_broadcast_stress_case_038() {
        let t = Tensor::full(vec![1, 2], 38.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 38.0);
        assert_eq!(b.get(5), 38.0);
    }

    #[test]
    fn test_broadcast_stress_case_039() {
        let t = Tensor::full(vec![1, 2], 39.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 39.0);
        assert_eq!(b.get(5), 39.0);
    }

    #[test]
    fn test_broadcast_stress_case_040() {
        let t = Tensor::full(vec![1, 2], 40.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 40.0);
        assert_eq!(b.get(5), 40.0);
    }

    #[test]
    fn test_broadcast_stress_case_041() {
        let t = Tensor::full(vec![1, 2], 41.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 41.0);
        assert_eq!(b.get(5), 41.0);
    }

    #[test]
    fn test_broadcast_stress_case_042() {
        let t = Tensor::full(vec![1, 2], 42.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 42.0);
        assert_eq!(b.get(5), 42.0);
    }

    #[test]
    fn test_broadcast_stress_case_043() {
        let t = Tensor::full(vec![1, 2], 43.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 43.0);
        assert_eq!(b.get(5), 43.0);
    }

    #[test]
    fn test_broadcast_stress_case_044() {
        let t = Tensor::full(vec![1, 2], 44.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 44.0);
        assert_eq!(b.get(5), 44.0);
    }

    #[test]
    fn test_broadcast_stress_case_045() {
        let t = Tensor::full(vec![1, 2], 45.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 45.0);
        assert_eq!(b.get(5), 45.0);
    }

    #[test]
    fn test_broadcast_stress_case_046() {
        let t = Tensor::full(vec![1, 2], 46.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 46.0);
        assert_eq!(b.get(5), 46.0);
    }

    #[test]
    fn test_broadcast_stress_case_047() {
        let t = Tensor::full(vec![1, 2], 47.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 47.0);
        assert_eq!(b.get(5), 47.0);
    }

    #[test]
    fn test_broadcast_stress_case_048() {
        let t = Tensor::full(vec![1, 2], 48.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 48.0);
        assert_eq!(b.get(5), 48.0);
    }

    #[test]
    fn test_broadcast_stress_case_049() {
        let t = Tensor::full(vec![1, 2], 49.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 49.0);
        assert_eq!(b.get(5), 49.0);
    }

    #[test]
    fn test_broadcast_stress_case_050() {
        let t = Tensor::full(vec![1, 2], 50.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 50.0);
        assert_eq!(b.get(5), 50.0);
    }

    #[test]
    fn test_broadcast_stress_case_051() {
        let t = Tensor::full(vec![1, 2], 51.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 51.0);
        assert_eq!(b.get(5), 51.0);
    }

    #[test]
    fn test_broadcast_stress_case_052() {
        let t = Tensor::full(vec![1, 2], 52.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 52.0);
        assert_eq!(b.get(5), 52.0);
    }

    #[test]
    fn test_broadcast_stress_case_053() {
        let t = Tensor::full(vec![1, 2], 53.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 53.0);
        assert_eq!(b.get(5), 53.0);
    }

    #[test]
    fn test_broadcast_stress_case_054() {
        let t = Tensor::full(vec![1, 2], 54.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 54.0);
        assert_eq!(b.get(5), 54.0);
    }

    #[test]
    fn test_broadcast_stress_case_055() {
        let t = Tensor::full(vec![1, 2], 55.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 55.0);
        assert_eq!(b.get(5), 55.0);
    }

    #[test]
    fn test_broadcast_stress_case_056() {
        let t = Tensor::full(vec![1, 2], 56.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 56.0);
        assert_eq!(b.get(5), 56.0);
    }

    #[test]
    fn test_broadcast_stress_case_057() {
        let t = Tensor::full(vec![1, 2], 57.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 57.0);
        assert_eq!(b.get(5), 57.0);
    }

    #[test]
    fn test_broadcast_stress_case_058() {
        let t = Tensor::full(vec![1, 2], 58.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 58.0);
        assert_eq!(b.get(5), 58.0);
    }

    #[test]
    fn test_broadcast_stress_case_059() {
        let t = Tensor::full(vec![1, 2], 59.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 59.0);
        assert_eq!(b.get(5), 59.0);
    }

    #[test]
    fn test_broadcast_stress_case_060() {
        let t = Tensor::full(vec![1, 2], 60.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 60.0);
        assert_eq!(b.get(5), 60.0);
    }

    #[test]
    fn test_broadcast_stress_case_061() {
        let t = Tensor::full(vec![1, 2], 61.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 61.0);
        assert_eq!(b.get(5), 61.0);
    }

    #[test]
    fn test_broadcast_stress_case_062() {
        let t = Tensor::full(vec![1, 2], 62.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 62.0);
        assert_eq!(b.get(5), 62.0);
    }

    #[test]
    fn test_broadcast_stress_case_063() {
        let t = Tensor::full(vec![1, 2], 63.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 63.0);
        assert_eq!(b.get(5), 63.0);
    }

    #[test]
    fn test_broadcast_stress_case_064() {
        let t = Tensor::full(vec![1, 2], 64.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 64.0);
        assert_eq!(b.get(5), 64.0);
    }

    #[test]
    fn test_broadcast_stress_case_065() {
        let t = Tensor::full(vec![1, 2], 65.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 65.0);
        assert_eq!(b.get(5), 65.0);
    }

    #[test]
    fn test_broadcast_stress_case_066() {
        let t = Tensor::full(vec![1, 2], 66.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 66.0);
        assert_eq!(b.get(5), 66.0);
    }

    #[test]
    fn test_broadcast_stress_case_067() {
        let t = Tensor::full(vec![1, 2], 67.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 67.0);
        assert_eq!(b.get(5), 67.0);
    }

    #[test]
    fn test_broadcast_stress_case_068() {
        let t = Tensor::full(vec![1, 2], 68.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 68.0);
        assert_eq!(b.get(5), 68.0);
    }

    #[test]
    fn test_broadcast_stress_case_069() {
        let t = Tensor::full(vec![1, 2], 69.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 69.0);
        assert_eq!(b.get(5), 69.0);
    }

    #[test]
    fn test_broadcast_stress_case_070() {
        let t = Tensor::full(vec![1, 2], 70.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 70.0);
        assert_eq!(b.get(5), 70.0);
    }

    #[test]
    fn test_broadcast_stress_case_071() {
        let t = Tensor::full(vec![1, 2], 71.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 71.0);
        assert_eq!(b.get(5), 71.0);
    }

    #[test]
    fn test_broadcast_stress_case_072() {
        let t = Tensor::full(vec![1, 2], 72.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 72.0);
        assert_eq!(b.get(5), 72.0);
    }

    #[test]
    fn test_broadcast_stress_case_073() {
        let t = Tensor::full(vec![1, 2], 73.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 73.0);
        assert_eq!(b.get(5), 73.0);
    }

    #[test]
    fn test_broadcast_stress_case_074() {
        let t = Tensor::full(vec![1, 2], 74.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 74.0);
        assert_eq!(b.get(5), 74.0);
    }

    #[test]
    fn test_broadcast_stress_case_075() {
        let t = Tensor::full(vec![1, 2], 75.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 75.0);
        assert_eq!(b.get(5), 75.0);
    }

    #[test]
    fn test_broadcast_stress_case_076() {
        let t = Tensor::full(vec![1, 2], 76.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 76.0);
        assert_eq!(b.get(5), 76.0);
    }

    #[test]
    fn test_broadcast_stress_case_077() {
        let t = Tensor::full(vec![1, 2], 77.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 77.0);
        assert_eq!(b.get(5), 77.0);
    }

    #[test]
    fn test_broadcast_stress_case_078() {
        let t = Tensor::full(vec![1, 2], 78.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 78.0);
        assert_eq!(b.get(5), 78.0);
    }

    #[test]
    fn test_broadcast_stress_case_079() {
        let t = Tensor::full(vec![1, 2], 79.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 79.0);
        assert_eq!(b.get(5), 79.0);
    }

    #[test]
    fn test_broadcast_stress_case_080() {
        let t = Tensor::full(vec![1, 2], 80.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 80.0);
        assert_eq!(b.get(5), 80.0);
    }

    #[test]
    fn test_broadcast_stress_case_081() {
        let t = Tensor::full(vec![1, 2], 81.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 81.0);
        assert_eq!(b.get(5), 81.0);
    }

    #[test]
    fn test_broadcast_stress_case_082() {
        let t = Tensor::full(vec![1, 2], 82.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 82.0);
        assert_eq!(b.get(5), 82.0);
    }

    #[test]
    fn test_broadcast_stress_case_083() {
        let t = Tensor::full(vec![1, 2], 83.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 83.0);
        assert_eq!(b.get(5), 83.0);
    }

    #[test]
    fn test_broadcast_stress_case_084() {
        let t = Tensor::full(vec![1, 2], 84.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 84.0);
        assert_eq!(b.get(5), 84.0);
    }

    #[test]
    fn test_broadcast_stress_case_085() {
        let t = Tensor::full(vec![1, 2], 85.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 85.0);
        assert_eq!(b.get(5), 85.0);
    }

    #[test]
    fn test_broadcast_stress_case_086() {
        let t = Tensor::full(vec![1, 2], 86.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 86.0);
        assert_eq!(b.get(5), 86.0);
    }

    #[test]
    fn test_broadcast_stress_case_087() {
        let t = Tensor::full(vec![1, 2], 87.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 87.0);
        assert_eq!(b.get(5), 87.0);
    }

    #[test]
    fn test_broadcast_stress_case_088() {
        let t = Tensor::full(vec![1, 2], 88.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 88.0);
        assert_eq!(b.get(5), 88.0);
    }

    #[test]
    fn test_broadcast_stress_case_089() {
        let t = Tensor::full(vec![1, 2], 89.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 89.0);
        assert_eq!(b.get(5), 89.0);
    }

    #[test]
    fn test_broadcast_stress_case_090() {
        let t = Tensor::full(vec![1, 2], 90.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 90.0);
        assert_eq!(b.get(5), 90.0);
    }

    #[test]
    fn test_broadcast_stress_case_091() {
        let t = Tensor::full(vec![1, 2], 91.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 91.0);
        assert_eq!(b.get(5), 91.0);
    }

    #[test]
    fn test_broadcast_stress_case_092() {
        let t = Tensor::full(vec![1, 2], 92.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 92.0);
        assert_eq!(b.get(5), 92.0);
    }

    #[test]
    fn test_broadcast_stress_case_093() {
        let t = Tensor::full(vec![1, 2], 93.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 93.0);
        assert_eq!(b.get(5), 93.0);
    }

    #[test]
    fn test_broadcast_stress_case_094() {
        let t = Tensor::full(vec![1, 2], 94.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 94.0);
        assert_eq!(b.get(5), 94.0);
    }

    #[test]
    fn test_broadcast_stress_case_095() {
        let t = Tensor::full(vec![1, 2], 95.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 95.0);
        assert_eq!(b.get(5), 95.0);
    }

    #[test]
    fn test_broadcast_stress_case_096() {
        let t = Tensor::full(vec![1, 2], 96.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 96.0);
        assert_eq!(b.get(5), 96.0);
    }

    #[test]
    fn test_broadcast_stress_case_097() {
        let t = Tensor::full(vec![1, 2], 97.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 97.0);
        assert_eq!(b.get(5), 97.0);
    }

    #[test]
    fn test_broadcast_stress_case_098() {
        let t = Tensor::full(vec![1, 2], 98.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 98.0);
        assert_eq!(b.get(5), 98.0);
    }

    #[test]
    fn test_broadcast_stress_case_099() {
        let t = Tensor::full(vec![1, 2], 99.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 99.0);
        assert_eq!(b.get(5), 99.0);
    }

    #[test]
    fn test_broadcast_stress_case_100() {
        let t = Tensor::full(vec![1, 2], 100.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 100.0);
        assert_eq!(b.get(5), 100.0);
    }

    #[test]
    fn test_broadcast_stress_case_101() {
        let t = Tensor::full(vec![1, 2], 101.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 101.0);
        assert_eq!(b.get(5), 101.0);
    }

    #[test]
    fn test_broadcast_stress_case_102() {
        let t = Tensor::full(vec![1, 2], 102.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 102.0);
        assert_eq!(b.get(5), 102.0);
    }

    #[test]
    fn test_broadcast_stress_case_103() {
        let t = Tensor::full(vec![1, 2], 103.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 103.0);
        assert_eq!(b.get(5), 103.0);
    }

    #[test]
    fn test_broadcast_stress_case_104() {
        let t = Tensor::full(vec![1, 2], 104.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 104.0);
        assert_eq!(b.get(5), 104.0);
    }

    #[test]
    fn test_broadcast_stress_case_105() {
        let t = Tensor::full(vec![1, 2], 105.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 105.0);
        assert_eq!(b.get(5), 105.0);
    }

    #[test]
    fn test_broadcast_stress_case_106() {
        let t = Tensor::full(vec![1, 2], 106.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 106.0);
        assert_eq!(b.get(5), 106.0);
    }

    #[test]
    fn test_broadcast_stress_case_107() {
        let t = Tensor::full(vec![1, 2], 107.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 107.0);
        assert_eq!(b.get(5), 107.0);
    }

    #[test]
    fn test_broadcast_stress_case_108() {
        let t = Tensor::full(vec![1, 2], 108.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 108.0);
        assert_eq!(b.get(5), 108.0);
    }

    #[test]
    fn test_broadcast_stress_case_109() {
        let t = Tensor::full(vec![1, 2], 109.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 109.0);
        assert_eq!(b.get(5), 109.0);
    }

    #[test]
    fn test_broadcast_stress_case_110() {
        let t = Tensor::full(vec![1, 2], 110.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 110.0);
        assert_eq!(b.get(5), 110.0);
    }

    #[test]
    fn test_broadcast_stress_case_111() {
        let t = Tensor::full(vec![1, 2], 111.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 111.0);
        assert_eq!(b.get(5), 111.0);
    }

    #[test]
    fn test_broadcast_stress_case_112() {
        let t = Tensor::full(vec![1, 2], 112.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 112.0);
        assert_eq!(b.get(5), 112.0);
    }

    #[test]
    fn test_broadcast_stress_case_113() {
        let t = Tensor::full(vec![1, 2], 113.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 113.0);
        assert_eq!(b.get(5), 113.0);
    }

    #[test]
    fn test_broadcast_stress_case_114() {
        let t = Tensor::full(vec![1, 2], 114.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 114.0);
        assert_eq!(b.get(5), 114.0);
    }

    #[test]
    fn test_broadcast_stress_case_115() {
        let t = Tensor::full(vec![1, 2], 115.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 115.0);
        assert_eq!(b.get(5), 115.0);
    }

    #[test]
    fn test_broadcast_stress_case_116() {
        let t = Tensor::full(vec![1, 2], 116.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 116.0);
        assert_eq!(b.get(5), 116.0);
    }

    #[test]
    fn test_broadcast_stress_case_117() {
        let t = Tensor::full(vec![1, 2], 117.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 117.0);
        assert_eq!(b.get(5), 117.0);
    }

    #[test]
    fn test_broadcast_stress_case_118() {
        let t = Tensor::full(vec![1, 2], 118.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 118.0);
        assert_eq!(b.get(5), 118.0);
    }

    #[test]
    fn test_broadcast_stress_case_119() {
        let t = Tensor::full(vec![1, 2], 119.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 119.0);
        assert_eq!(b.get(5), 119.0);
    }

    #[test]
    fn test_broadcast_stress_case_120() {
        let t = Tensor::full(vec![1, 2], 120.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 120.0);
        assert_eq!(b.get(5), 120.0);
    }

    #[test]
    fn test_broadcast_stress_case_121() {
        let t = Tensor::full(vec![1, 2], 121.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 121.0);
        assert_eq!(b.get(5), 121.0);
    }

    #[test]
    fn test_broadcast_stress_case_122() {
        let t = Tensor::full(vec![1, 2], 122.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 122.0);
        assert_eq!(b.get(5), 122.0);
    }

    #[test]
    fn test_broadcast_stress_case_123() {
        let t = Tensor::full(vec![1, 2], 123.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 123.0);
        assert_eq!(b.get(5), 123.0);
    }

    #[test]
    fn test_broadcast_stress_case_124() {
        let t = Tensor::full(vec![1, 2], 124.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 124.0);
        assert_eq!(b.get(5), 124.0);
    }

    #[test]
    fn test_broadcast_stress_case_125() {
        let t = Tensor::full(vec![1, 2], 125.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 125.0);
        assert_eq!(b.get(5), 125.0);
    }

    #[test]
    fn test_broadcast_stress_case_126() {
        let t = Tensor::full(vec![1, 2], 126.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 126.0);
        assert_eq!(b.get(5), 126.0);
    }

    #[test]
    fn test_broadcast_stress_case_127() {
        let t = Tensor::full(vec![1, 2], 127.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 127.0);
        assert_eq!(b.get(5), 127.0);
    }

    #[test]
    fn test_broadcast_stress_case_128() {
        let t = Tensor::full(vec![1, 2], 128.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 128.0);
        assert_eq!(b.get(5), 128.0);
    }

    #[test]
    fn test_broadcast_stress_case_129() {
        let t = Tensor::full(vec![1, 2], 129.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 129.0);
        assert_eq!(b.get(5), 129.0);
    }

    #[test]
    fn test_broadcast_stress_case_130() {
        let t = Tensor::full(vec![1, 2], 130.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 130.0);
        assert_eq!(b.get(5), 130.0);
    }

    #[test]
    fn test_broadcast_stress_case_131() {
        let t = Tensor::full(vec![1, 2], 131.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 131.0);
        assert_eq!(b.get(5), 131.0);
    }

    #[test]
    fn test_broadcast_stress_case_132() {
        let t = Tensor::full(vec![1, 2], 132.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 132.0);
        assert_eq!(b.get(5), 132.0);
    }

    #[test]
    fn test_broadcast_stress_case_133() {
        let t = Tensor::full(vec![1, 2], 133.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 133.0);
        assert_eq!(b.get(5), 133.0);
    }

    #[test]
    fn test_broadcast_stress_case_134() {
        let t = Tensor::full(vec![1, 2], 134.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 134.0);
        assert_eq!(b.get(5), 134.0);
    }

    #[test]
    fn test_broadcast_stress_case_135() {
        let t = Tensor::full(vec![1, 2], 135.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 135.0);
        assert_eq!(b.get(5), 135.0);
    }

    #[test]
    fn test_broadcast_stress_case_136() {
        let t = Tensor::full(vec![1, 2], 136.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 136.0);
        assert_eq!(b.get(5), 136.0);
    }

    #[test]
    fn test_broadcast_stress_case_137() {
        let t = Tensor::full(vec![1, 2], 137.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 137.0);
        assert_eq!(b.get(5), 137.0);
    }

    #[test]
    fn test_broadcast_stress_case_138() {
        let t = Tensor::full(vec![1, 2], 138.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 138.0);
        assert_eq!(b.get(5), 138.0);
    }

    #[test]
    fn test_broadcast_stress_case_139() {
        let t = Tensor::full(vec![1, 2], 139.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 139.0);
        assert_eq!(b.get(5), 139.0);
    }

    #[test]
    fn test_broadcast_stress_case_140() {
        let t = Tensor::full(vec![1, 2], 140.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 140.0);
        assert_eq!(b.get(5), 140.0);
    }

    #[test]
    fn test_broadcast_stress_case_141() {
        let t = Tensor::full(vec![1, 2], 141.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 141.0);
        assert_eq!(b.get(5), 141.0);
    }

    #[test]
    fn test_broadcast_stress_case_142() {
        let t = Tensor::full(vec![1, 2], 142.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 142.0);
        assert_eq!(b.get(5), 142.0);
    }

    #[test]
    fn test_broadcast_stress_case_143() {
        let t = Tensor::full(vec![1, 2], 143.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 143.0);
        assert_eq!(b.get(5), 143.0);
    }

    #[test]
    fn test_broadcast_stress_case_144() {
        let t = Tensor::full(vec![1, 2], 144.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 144.0);
        assert_eq!(b.get(5), 144.0);
    }

    #[test]
    fn test_broadcast_stress_case_145() {
        let t = Tensor::full(vec![1, 2], 145.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 145.0);
        assert_eq!(b.get(5), 145.0);
    }

    #[test]
    fn test_broadcast_stress_case_146() {
        let t = Tensor::full(vec![1, 2], 146.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 146.0);
        assert_eq!(b.get(5), 146.0);
    }

    #[test]
    fn test_broadcast_stress_case_147() {
        let t = Tensor::full(vec![1, 2], 147.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 147.0);
        assert_eq!(b.get(5), 147.0);
    }

    #[test]
    fn test_broadcast_stress_case_148() {
        let t = Tensor::full(vec![1, 2], 148.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 148.0);
        assert_eq!(b.get(5), 148.0);
    }

    #[test]
    fn test_broadcast_stress_case_149() {
        let t = Tensor::full(vec![1, 2], 149.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 149.0);
        assert_eq!(b.get(5), 149.0);
    }

    #[test]
    fn test_broadcast_stress_case_150() {
        let t = Tensor::full(vec![1, 2], 150.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 150.0);
        assert_eq!(b.get(5), 150.0);
    }

    #[test]
    fn test_broadcast_stress_case_151() {
        let t = Tensor::full(vec![1, 2], 151.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 151.0);
        assert_eq!(b.get(5), 151.0);
    }

    #[test]
    fn test_broadcast_stress_case_152() {
        let t = Tensor::full(vec![1, 2], 152.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 152.0);
        assert_eq!(b.get(5), 152.0);
    }

    #[test]
    fn test_broadcast_stress_case_153() {
        let t = Tensor::full(vec![1, 2], 153.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 153.0);
        assert_eq!(b.get(5), 153.0);
    }

    #[test]
    fn test_broadcast_stress_case_154() {
        let t = Tensor::full(vec![1, 2], 154.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 154.0);
        assert_eq!(b.get(5), 154.0);
    }

    #[test]
    fn test_broadcast_stress_case_155() {
        let t = Tensor::full(vec![1, 2], 155.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 155.0);
        assert_eq!(b.get(5), 155.0);
    }

    #[test]
    fn test_broadcast_stress_case_156() {
        let t = Tensor::full(vec![1, 2], 156.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 156.0);
        assert_eq!(b.get(5), 156.0);
    }

    #[test]
    fn test_broadcast_stress_case_157() {
        let t = Tensor::full(vec![1, 2], 157.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 157.0);
        assert_eq!(b.get(5), 157.0);
    }

    #[test]
    fn test_broadcast_stress_case_158() {
        let t = Tensor::full(vec![1, 2], 158.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 158.0);
        assert_eq!(b.get(5), 158.0);
    }

    #[test]
    fn test_broadcast_stress_case_159() {
        let t = Tensor::full(vec![1, 2], 159.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 159.0);
        assert_eq!(b.get(5), 159.0);
    }

    #[test]
    fn test_broadcast_stress_case_160() {
        let t = Tensor::full(vec![1, 2], 160.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 160.0);
        assert_eq!(b.get(5), 160.0);
    }

    #[test]
    fn test_broadcast_stress_case_161() {
        let t = Tensor::full(vec![1, 2], 161.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 161.0);
        assert_eq!(b.get(5), 161.0);
    }

    #[test]
    fn test_broadcast_stress_case_162() {
        let t = Tensor::full(vec![1, 2], 162.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 162.0);
        assert_eq!(b.get(5), 162.0);
    }

    #[test]
    fn test_broadcast_stress_case_163() {
        let t = Tensor::full(vec![1, 2], 163.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 163.0);
        assert_eq!(b.get(5), 163.0);
    }

    #[test]
    fn test_broadcast_stress_case_164() {
        let t = Tensor::full(vec![1, 2], 164.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 164.0);
        assert_eq!(b.get(5), 164.0);
    }

    #[test]
    fn test_broadcast_stress_case_165() {
        let t = Tensor::full(vec![1, 2], 165.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 165.0);
        assert_eq!(b.get(5), 165.0);
    }

    #[test]
    fn test_broadcast_stress_case_166() {
        let t = Tensor::full(vec![1, 2], 166.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 166.0);
        assert_eq!(b.get(5), 166.0);
    }

    #[test]
    fn test_broadcast_stress_case_167() {
        let t = Tensor::full(vec![1, 2], 167.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 167.0);
        assert_eq!(b.get(5), 167.0);
    }

    #[test]
    fn test_broadcast_stress_case_168() {
        let t = Tensor::full(vec![1, 2], 168.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 168.0);
        assert_eq!(b.get(5), 168.0);
    }

    #[test]
    fn test_broadcast_stress_case_169() {
        let t = Tensor::full(vec![1, 2], 169.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 169.0);
        assert_eq!(b.get(5), 169.0);
    }

    #[test]
    fn test_broadcast_stress_case_170() {
        let t = Tensor::full(vec![1, 2], 170.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 170.0);
        assert_eq!(b.get(5), 170.0);
    }

    #[test]
    fn test_broadcast_stress_case_171() {
        let t = Tensor::full(vec![1, 2], 171.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 171.0);
        assert_eq!(b.get(5), 171.0);
    }

    #[test]
    fn test_broadcast_stress_case_172() {
        let t = Tensor::full(vec![1, 2], 172.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 172.0);
        assert_eq!(b.get(5), 172.0);
    }

    #[test]
    fn test_broadcast_stress_case_173() {
        let t = Tensor::full(vec![1, 2], 173.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 173.0);
        assert_eq!(b.get(5), 173.0);
    }

    #[test]
    fn test_broadcast_stress_case_174() {
        let t = Tensor::full(vec![1, 2], 174.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 174.0);
        assert_eq!(b.get(5), 174.0);
    }

    #[test]
    fn test_broadcast_stress_case_175() {
        let t = Tensor::full(vec![1, 2], 175.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 175.0);
        assert_eq!(b.get(5), 175.0);
    }

    #[test]
    fn test_broadcast_stress_case_176() {
        let t = Tensor::full(vec![1, 2], 176.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 176.0);
        assert_eq!(b.get(5), 176.0);
    }

    #[test]
    fn test_broadcast_stress_case_177() {
        let t = Tensor::full(vec![1, 2], 177.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 177.0);
        assert_eq!(b.get(5), 177.0);
    }

    #[test]
    fn test_broadcast_stress_case_178() {
        let t = Tensor::full(vec![1, 2], 178.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 178.0);
        assert_eq!(b.get(5), 178.0);
    }

    #[test]
    fn test_broadcast_stress_case_179() {
        let t = Tensor::full(vec![1, 2], 179.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 179.0);
        assert_eq!(b.get(5), 179.0);
    }

    #[test]
    fn test_broadcast_stress_case_180() {
        let t = Tensor::full(vec![1, 2], 180.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 180.0);
        assert_eq!(b.get(5), 180.0);
    }

    #[test]
    fn test_broadcast_stress_case_181() {
        let t = Tensor::full(vec![1, 2], 181.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 181.0);
        assert_eq!(b.get(5), 181.0);
    }

    #[test]
    fn test_broadcast_stress_case_182() {
        let t = Tensor::full(vec![1, 2], 182.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 182.0);
        assert_eq!(b.get(5), 182.0);
    }

    #[test]
    fn test_broadcast_stress_case_183() {
        let t = Tensor::full(vec![1, 2], 183.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 183.0);
        assert_eq!(b.get(5), 183.0);
    }

    #[test]
    fn test_broadcast_stress_case_184() {
        let t = Tensor::full(vec![1, 2], 184.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 184.0);
        assert_eq!(b.get(5), 184.0);
    }

    #[test]
    fn test_broadcast_stress_case_185() {
        let t = Tensor::full(vec![1, 2], 185.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 185.0);
        assert_eq!(b.get(5), 185.0);
    }

    #[test]
    fn test_broadcast_stress_case_186() {
        let t = Tensor::full(vec![1, 2], 186.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 186.0);
        assert_eq!(b.get(5), 186.0);
    }

    #[test]
    fn test_broadcast_stress_case_187() {
        let t = Tensor::full(vec![1, 2], 187.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 187.0);
        assert_eq!(b.get(5), 187.0);
    }

    #[test]
    fn test_broadcast_stress_case_188() {
        let t = Tensor::full(vec![1, 2], 188.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 188.0);
        assert_eq!(b.get(5), 188.0);
    }

    #[test]
    fn test_broadcast_stress_case_189() {
        let t = Tensor::full(vec![1, 2], 189.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 189.0);
        assert_eq!(b.get(5), 189.0);
    }

    #[test]
    fn test_broadcast_stress_case_190() {
        let t = Tensor::full(vec![1, 2], 190.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 190.0);
        assert_eq!(b.get(5), 190.0);
    }

    #[test]
    fn test_broadcast_stress_case_191() {
        let t = Tensor::full(vec![1, 2], 191.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 191.0);
        assert_eq!(b.get(5), 191.0);
    }

    #[test]
    fn test_broadcast_stress_case_192() {
        let t = Tensor::full(vec![1, 2], 192.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 192.0);
        assert_eq!(b.get(5), 192.0);
    }

    #[test]
    fn test_broadcast_stress_case_193() {
        let t = Tensor::full(vec![1, 2], 193.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 193.0);
        assert_eq!(b.get(5), 193.0);
    }

    #[test]
    fn test_broadcast_stress_case_194() {
        let t = Tensor::full(vec![1, 2], 194.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 194.0);
        assert_eq!(b.get(5), 194.0);
    }

    #[test]
    fn test_broadcast_stress_case_195() {
        let t = Tensor::full(vec![1, 2], 195.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 195.0);
        assert_eq!(b.get(5), 195.0);
    }

    #[test]
    fn test_broadcast_stress_case_196() {
        let t = Tensor::full(vec![1, 2], 196.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 196.0);
        assert_eq!(b.get(5), 196.0);
    }

    #[test]
    fn test_broadcast_stress_case_197() {
        let t = Tensor::full(vec![1, 2], 197.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 197.0);
        assert_eq!(b.get(5), 197.0);
    }

    #[test]
    fn test_broadcast_stress_case_198() {
        let t = Tensor::full(vec![1, 2], 198.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 198.0);
        assert_eq!(b.get(5), 198.0);
    }

    #[test]
    fn test_broadcast_stress_case_199() {
        let t = Tensor::full(vec![1, 2], 199.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 199.0);
        assert_eq!(b.get(5), 199.0);
    }

    #[test]
    fn test_broadcast_stress_case_200() {
        let t = Tensor::full(vec![1, 2], 200.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 200.0);
        assert_eq!(b.get(5), 200.0);
    }

    #[test]
    fn test_broadcast_stress_case_201() {
        let t = Tensor::full(vec![1, 2], 201.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 201.0);
        assert_eq!(b.get(5), 201.0);
    }

    #[test]
    fn test_broadcast_stress_case_202() {
        let t = Tensor::full(vec![1, 2], 202.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 202.0);
        assert_eq!(b.get(5), 202.0);
    }

    #[test]
    fn test_broadcast_stress_case_203() {
        let t = Tensor::full(vec![1, 2], 203.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 203.0);
        assert_eq!(b.get(5), 203.0);
    }

    #[test]
    fn test_broadcast_stress_case_204() {
        let t = Tensor::full(vec![1, 2], 204.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 204.0);
        assert_eq!(b.get(5), 204.0);
    }

    #[test]
    fn test_broadcast_stress_case_205() {
        let t = Tensor::full(vec![1, 2], 205.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 205.0);
        assert_eq!(b.get(5), 205.0);
    }

    #[test]
    fn test_broadcast_stress_case_206() {
        let t = Tensor::full(vec![1, 2], 206.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 206.0);
        assert_eq!(b.get(5), 206.0);
    }

    #[test]
    fn test_broadcast_stress_case_207() {
        let t = Tensor::full(vec![1, 2], 207.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 207.0);
        assert_eq!(b.get(5), 207.0);
    }

    #[test]
    fn test_broadcast_stress_case_208() {
        let t = Tensor::full(vec![1, 2], 208.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 208.0);
        assert_eq!(b.get(5), 208.0);
    }

    #[test]
    fn test_broadcast_stress_case_209() {
        let t = Tensor::full(vec![1, 2], 209.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 209.0);
        assert_eq!(b.get(5), 209.0);
    }

    #[test]
    fn test_broadcast_stress_case_210() {
        let t = Tensor::full(vec![1, 2], 210.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 210.0);
        assert_eq!(b.get(5), 210.0);
    }

    #[test]
    fn test_broadcast_stress_case_211() {
        let t = Tensor::full(vec![1, 2], 211.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 211.0);
        assert_eq!(b.get(5), 211.0);
    }

    #[test]
    fn test_broadcast_stress_case_212() {
        let t = Tensor::full(vec![1, 2], 212.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 212.0);
        assert_eq!(b.get(5), 212.0);
    }

    #[test]
    fn test_broadcast_stress_case_213() {
        let t = Tensor::full(vec![1, 2], 213.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 213.0);
        assert_eq!(b.get(5), 213.0);
    }

    #[test]
    fn test_broadcast_stress_case_214() {
        let t = Tensor::full(vec![1, 2], 214.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 214.0);
        assert_eq!(b.get(5), 214.0);
    }

    #[test]
    fn test_broadcast_stress_case_215() {
        let t = Tensor::full(vec![1, 2], 215.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 215.0);
        assert_eq!(b.get(5), 215.0);
    }

    #[test]
    fn test_broadcast_stress_case_216() {
        let t = Tensor::full(vec![1, 2], 216.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 216.0);
        assert_eq!(b.get(5), 216.0);
    }

    #[test]
    fn test_broadcast_stress_case_217() {
        let t = Tensor::full(vec![1, 2], 217.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 217.0);
        assert_eq!(b.get(5), 217.0);
    }

    #[test]
    fn test_broadcast_stress_case_218() {
        let t = Tensor::full(vec![1, 2], 218.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 218.0);
        assert_eq!(b.get(5), 218.0);
    }

    #[test]
    fn test_broadcast_stress_case_219() {
        let t = Tensor::full(vec![1, 2], 219.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 219.0);
        assert_eq!(b.get(5), 219.0);
    }

    #[test]
    fn test_broadcast_stress_case_220() {
        let t = Tensor::full(vec![1, 2], 220.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 220.0);
        assert_eq!(b.get(5), 220.0);
    }

    #[test]
    fn test_broadcast_stress_case_221() {
        let t = Tensor::full(vec![1, 2], 221.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 221.0);
        assert_eq!(b.get(5), 221.0);
    }

    #[test]
    fn test_broadcast_stress_case_222() {
        let t = Tensor::full(vec![1, 2], 222.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 222.0);
        assert_eq!(b.get(5), 222.0);
    }

    #[test]
    fn test_broadcast_stress_case_223() {
        let t = Tensor::full(vec![1, 2], 223.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 223.0);
        assert_eq!(b.get(5), 223.0);
    }

    #[test]
    fn test_broadcast_stress_case_224() {
        let t = Tensor::full(vec![1, 2], 224.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 224.0);
        assert_eq!(b.get(5), 224.0);
    }

    #[test]
    fn test_broadcast_stress_case_225() {
        let t = Tensor::full(vec![1, 2], 225.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 225.0);
        assert_eq!(b.get(5), 225.0);
    }

    #[test]
    fn test_broadcast_stress_case_226() {
        let t = Tensor::full(vec![1, 2], 226.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 226.0);
        assert_eq!(b.get(5), 226.0);
    }

    #[test]
    fn test_broadcast_stress_case_227() {
        let t = Tensor::full(vec![1, 2], 227.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 227.0);
        assert_eq!(b.get(5), 227.0);
    }

    #[test]
    fn test_broadcast_stress_case_228() {
        let t = Tensor::full(vec![1, 2], 228.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 228.0);
        assert_eq!(b.get(5), 228.0);
    }

    #[test]
    fn test_broadcast_stress_case_229() {
        let t = Tensor::full(vec![1, 2], 229.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 229.0);
        assert_eq!(b.get(5), 229.0);
    }

    #[test]
    fn test_broadcast_stress_case_230() {
        let t = Tensor::full(vec![1, 2], 230.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 230.0);
        assert_eq!(b.get(5), 230.0);
    }

    #[test]
    fn test_broadcast_stress_case_231() {
        let t = Tensor::full(vec![1, 2], 231.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 231.0);
        assert_eq!(b.get(5), 231.0);
    }

    #[test]
    fn test_broadcast_stress_case_232() {
        let t = Tensor::full(vec![1, 2], 232.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 232.0);
        assert_eq!(b.get(5), 232.0);
    }

    #[test]
    fn test_broadcast_stress_case_233() {
        let t = Tensor::full(vec![1, 2], 233.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 233.0);
        assert_eq!(b.get(5), 233.0);
    }

    #[test]
    fn test_broadcast_stress_case_234() {
        let t = Tensor::full(vec![1, 2], 234.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 234.0);
        assert_eq!(b.get(5), 234.0);
    }

    #[test]
    fn test_broadcast_stress_case_235() {
        let t = Tensor::full(vec![1, 2], 235.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 235.0);
        assert_eq!(b.get(5), 235.0);
    }

    #[test]
    fn test_broadcast_stress_case_236() {
        let t = Tensor::full(vec![1, 2], 236.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 236.0);
        assert_eq!(b.get(5), 236.0);
    }

    #[test]
    fn test_broadcast_stress_case_237() {
        let t = Tensor::full(vec![1, 2], 237.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 237.0);
        assert_eq!(b.get(5), 237.0);
    }

    #[test]
    fn test_broadcast_stress_case_238() {
        let t = Tensor::full(vec![1, 2], 238.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 238.0);
        assert_eq!(b.get(5), 238.0);
    }

    #[test]
    fn test_broadcast_stress_case_239() {
        let t = Tensor::full(vec![1, 2], 239.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 239.0);
        assert_eq!(b.get(5), 239.0);
    }

    #[test]
    fn test_broadcast_stress_case_240() {
        let t = Tensor::full(vec![1, 2], 240.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 240.0);
        assert_eq!(b.get(5), 240.0);
    }

    #[test]
    fn test_broadcast_stress_case_241() {
        let t = Tensor::full(vec![1, 2], 241.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 241.0);
        assert_eq!(b.get(5), 241.0);
    }

    #[test]
    fn test_broadcast_stress_case_242() {
        let t = Tensor::full(vec![1, 2], 242.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 242.0);
        assert_eq!(b.get(5), 242.0);
    }

    #[test]
    fn test_broadcast_stress_case_243() {
        let t = Tensor::full(vec![1, 2], 243.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 243.0);
        assert_eq!(b.get(5), 243.0);
    }

    #[test]
    fn test_broadcast_stress_case_244() {
        let t = Tensor::full(vec![1, 2], 244.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 244.0);
        assert_eq!(b.get(5), 244.0);
    }

    #[test]
    fn test_broadcast_stress_case_245() {
        let t = Tensor::full(vec![1, 2], 245.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 245.0);
        assert_eq!(b.get(5), 245.0);
    }

    #[test]
    fn test_broadcast_stress_case_246() {
        let t = Tensor::full(vec![1, 2], 246.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 246.0);
        assert_eq!(b.get(5), 246.0);
    }

    #[test]
    fn test_broadcast_stress_case_247() {
        let t = Tensor::full(vec![1, 2], 247.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 247.0);
        assert_eq!(b.get(5), 247.0);
    }

    #[test]
    fn test_broadcast_stress_case_248() {
        let t = Tensor::full(vec![1, 2], 248.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 248.0);
        assert_eq!(b.get(5), 248.0);
    }

    #[test]
    fn test_broadcast_stress_case_249() {
        let t = Tensor::full(vec![1, 2], 249.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 249.0);
        assert_eq!(b.get(5), 249.0);
    }

    #[test]
    fn test_broadcast_stress_case_250() {
        let t = Tensor::full(vec![1, 2], 250.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 250.0);
        assert_eq!(b.get(5), 250.0);
    }

    #[test]
    fn test_broadcast_stress_case_251() {
        let t = Tensor::full(vec![1, 2], 251.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 251.0);
        assert_eq!(b.get(5), 251.0);
    }

    #[test]
    fn test_broadcast_stress_case_252() {
        let t = Tensor::full(vec![1, 2], 252.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 252.0);
        assert_eq!(b.get(5), 252.0);
    }

    #[test]
    fn test_broadcast_stress_case_253() {
        let t = Tensor::full(vec![1, 2], 253.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 253.0);
        assert_eq!(b.get(5), 253.0);
    }

    #[test]
    fn test_broadcast_stress_case_254() {
        let t = Tensor::full(vec![1, 2], 254.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 254.0);
        assert_eq!(b.get(5), 254.0);
    }

    #[test]
    fn test_broadcast_stress_case_255() {
        let t = Tensor::full(vec![1, 2], 255.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 255.0);
        assert_eq!(b.get(5), 255.0);
    }

    #[test]
    fn test_broadcast_stress_case_256() {
        let t = Tensor::full(vec![1, 2], 256.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 256.0);
        assert_eq!(b.get(5), 256.0);
    }

    #[test]
    fn test_broadcast_stress_case_257() {
        let t = Tensor::full(vec![1, 2], 257.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 257.0);
        assert_eq!(b.get(5), 257.0);
    }

    #[test]
    fn test_broadcast_stress_case_258() {
        let t = Tensor::full(vec![1, 2], 258.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 258.0);
        assert_eq!(b.get(5), 258.0);
    }

    #[test]
    fn test_broadcast_stress_case_259() {
        let t = Tensor::full(vec![1, 2], 259.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 259.0);
        assert_eq!(b.get(5), 259.0);
    }

    #[test]
    fn test_broadcast_stress_case_260() {
        let t = Tensor::full(vec![1, 2], 260.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 260.0);
        assert_eq!(b.get(5), 260.0);
    }

    #[test]
    fn test_broadcast_stress_case_261() {
        let t = Tensor::full(vec![1, 2], 261.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 261.0);
        assert_eq!(b.get(5), 261.0);
    }

    #[test]
    fn test_broadcast_stress_case_262() {
        let t = Tensor::full(vec![1, 2], 262.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 262.0);
        assert_eq!(b.get(5), 262.0);
    }

    #[test]
    fn test_broadcast_stress_case_263() {
        let t = Tensor::full(vec![1, 2], 263.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 263.0);
        assert_eq!(b.get(5), 263.0);
    }

    #[test]
    fn test_broadcast_stress_case_264() {
        let t = Tensor::full(vec![1, 2], 264.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 264.0);
        assert_eq!(b.get(5), 264.0);
    }

    #[test]
    fn test_broadcast_stress_case_265() {
        let t = Tensor::full(vec![1, 2], 265.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 265.0);
        assert_eq!(b.get(5), 265.0);
    }

    #[test]
    fn test_broadcast_stress_case_266() {
        let t = Tensor::full(vec![1, 2], 266.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 266.0);
        assert_eq!(b.get(5), 266.0);
    }

    #[test]
    fn test_broadcast_stress_case_267() {
        let t = Tensor::full(vec![1, 2], 267.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 267.0);
        assert_eq!(b.get(5), 267.0);
    }

    #[test]
    fn test_broadcast_stress_case_268() {
        let t = Tensor::full(vec![1, 2], 268.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 268.0);
        assert_eq!(b.get(5), 268.0);
    }

    #[test]
    fn test_broadcast_stress_case_269() {
        let t = Tensor::full(vec![1, 2], 269.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 269.0);
        assert_eq!(b.get(5), 269.0);
    }

    #[test]
    fn test_broadcast_stress_case_270() {
        let t = Tensor::full(vec![1, 2], 270.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 270.0);
        assert_eq!(b.get(5), 270.0);
    }

    #[test]
    fn test_broadcast_stress_case_271() {
        let t = Tensor::full(vec![1, 2], 271.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 271.0);
        assert_eq!(b.get(5), 271.0);
    }

    #[test]
    fn test_broadcast_stress_case_272() {
        let t = Tensor::full(vec![1, 2], 272.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 272.0);
        assert_eq!(b.get(5), 272.0);
    }

    #[test]
    fn test_broadcast_stress_case_273() {
        let t = Tensor::full(vec![1, 2], 273.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 273.0);
        assert_eq!(b.get(5), 273.0);
    }

    #[test]
    fn test_broadcast_stress_case_274() {
        let t = Tensor::full(vec![1, 2], 274.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 274.0);
        assert_eq!(b.get(5), 274.0);
    }

    #[test]
    fn test_broadcast_stress_case_275() {
        let t = Tensor::full(vec![1, 2], 275.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 275.0);
        assert_eq!(b.get(5), 275.0);
    }

    #[test]
    fn test_broadcast_stress_case_276() {
        let t = Tensor::full(vec![1, 2], 276.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 276.0);
        assert_eq!(b.get(5), 276.0);
    }

    #[test]
    fn test_broadcast_stress_case_277() {
        let t = Tensor::full(vec![1, 2], 277.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 277.0);
        assert_eq!(b.get(5), 277.0);
    }

    #[test]
    fn test_broadcast_stress_case_278() {
        let t = Tensor::full(vec![1, 2], 278.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 278.0);
        assert_eq!(b.get(5), 278.0);
    }

    #[test]
    fn test_broadcast_stress_case_279() {
        let t = Tensor::full(vec![1, 2], 279.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 279.0);
        assert_eq!(b.get(5), 279.0);
    }

    #[test]
    fn test_broadcast_stress_case_280() {
        let t = Tensor::full(vec![1, 2], 280.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 280.0);
        assert_eq!(b.get(5), 280.0);
    }

    #[test]
    fn test_broadcast_stress_case_281() {
        let t = Tensor::full(vec![1, 2], 281.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 281.0);
        assert_eq!(b.get(5), 281.0);
    }

    #[test]
    fn test_broadcast_stress_case_282() {
        let t = Tensor::full(vec![1, 2], 282.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 282.0);
        assert_eq!(b.get(5), 282.0);
    }

    #[test]
    fn test_broadcast_stress_case_283() {
        let t = Tensor::full(vec![1, 2], 283.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 283.0);
        assert_eq!(b.get(5), 283.0);
    }

    #[test]
    fn test_broadcast_stress_case_284() {
        let t = Tensor::full(vec![1, 2], 284.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 284.0);
        assert_eq!(b.get(5), 284.0);
    }

    #[test]
    fn test_broadcast_stress_case_285() {
        let t = Tensor::full(vec![1, 2], 285.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 285.0);
        assert_eq!(b.get(5), 285.0);
    }

    #[test]
    fn test_broadcast_stress_case_286() {
        let t = Tensor::full(vec![1, 2], 286.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 286.0);
        assert_eq!(b.get(5), 286.0);
    }

    #[test]
    fn test_broadcast_stress_case_287() {
        let t = Tensor::full(vec![1, 2], 287.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 287.0);
        assert_eq!(b.get(5), 287.0);
    }

    #[test]
    fn test_broadcast_stress_case_288() {
        let t = Tensor::full(vec![1, 2], 288.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 288.0);
        assert_eq!(b.get(5), 288.0);
    }

    #[test]
    fn test_broadcast_stress_case_289() {
        let t = Tensor::full(vec![1, 2], 289.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 289.0);
        assert_eq!(b.get(5), 289.0);
    }

    #[test]
    fn test_broadcast_stress_case_290() {
        let t = Tensor::full(vec![1, 2], 290.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 290.0);
        assert_eq!(b.get(5), 290.0);
    }

    #[test]
    fn test_broadcast_stress_case_291() {
        let t = Tensor::full(vec![1, 2], 291.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 291.0);
        assert_eq!(b.get(5), 291.0);
    }

    #[test]
    fn test_broadcast_stress_case_292() {
        let t = Tensor::full(vec![1, 2], 292.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 292.0);
        assert_eq!(b.get(5), 292.0);
    }

    #[test]
    fn test_broadcast_stress_case_293() {
        let t = Tensor::full(vec![1, 2], 293.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 293.0);
        assert_eq!(b.get(5), 293.0);
    }

    #[test]
    fn test_broadcast_stress_case_294() {
        let t = Tensor::full(vec![1, 2], 294.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 294.0);
        assert_eq!(b.get(5), 294.0);
    }

    #[test]
    fn test_broadcast_stress_case_295() {
        let t = Tensor::full(vec![1, 2], 295.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 295.0);
        assert_eq!(b.get(5), 295.0);
    }

    #[test]
    fn test_broadcast_stress_case_296() {
        let t = Tensor::full(vec![1, 2], 296.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 296.0);
        assert_eq!(b.get(5), 296.0);
    }

    #[test]
    fn test_broadcast_stress_case_297() {
        let t = Tensor::full(vec![1, 2], 297.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 297.0);
        assert_eq!(b.get(5), 297.0);
    }

    #[test]
    fn test_broadcast_stress_case_298() {
        let t = Tensor::full(vec![1, 2], 298.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 298.0);
        assert_eq!(b.get(5), 298.0);
    }

    #[test]
    fn test_broadcast_stress_case_299() {
        let t = Tensor::full(vec![1, 2], 299.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 299.0);
        assert_eq!(b.get(5), 299.0);
    }

    #[test]
    fn test_broadcast_stress_case_300() {
        let t = Tensor::full(vec![1, 2], 300.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 300.0);
        assert_eq!(b.get(5), 300.0);
    }

    #[test]
    fn test_broadcast_stress_case_301() {
        let t = Tensor::full(vec![1, 2], 301.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 301.0);
        assert_eq!(b.get(5), 301.0);
    }

    #[test]
    fn test_broadcast_stress_case_302() {
        let t = Tensor::full(vec![1, 2], 302.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 302.0);
        assert_eq!(b.get(5), 302.0);
    }

    #[test]
    fn test_broadcast_stress_case_303() {
        let t = Tensor::full(vec![1, 2], 303.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 303.0);
        assert_eq!(b.get(5), 303.0);
    }

    #[test]
    fn test_broadcast_stress_case_304() {
        let t = Tensor::full(vec![1, 2], 304.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 304.0);
        assert_eq!(b.get(5), 304.0);
    }

    #[test]
    fn test_broadcast_stress_case_305() {
        let t = Tensor::full(vec![1, 2], 305.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 305.0);
        assert_eq!(b.get(5), 305.0);
    }

    #[test]
    fn test_broadcast_stress_case_306() {
        let t = Tensor::full(vec![1, 2], 306.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 306.0);
        assert_eq!(b.get(5), 306.0);
    }

    #[test]
    fn test_broadcast_stress_case_307() {
        let t = Tensor::full(vec![1, 2], 307.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 307.0);
        assert_eq!(b.get(5), 307.0);
    }

    #[test]
    fn test_broadcast_stress_case_308() {
        let t = Tensor::full(vec![1, 2], 308.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 308.0);
        assert_eq!(b.get(5), 308.0);
    }

    #[test]
    fn test_broadcast_stress_case_309() {
        let t = Tensor::full(vec![1, 2], 309.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 309.0);
        assert_eq!(b.get(5), 309.0);
    }

    #[test]
    fn test_broadcast_stress_case_310() {
        let t = Tensor::full(vec![1, 2], 310.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 310.0);
        assert_eq!(b.get(5), 310.0);
    }

    #[test]
    fn test_broadcast_stress_case_311() {
        let t = Tensor::full(vec![1, 2], 311.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 311.0);
        assert_eq!(b.get(5), 311.0);
    }

    #[test]
    fn test_broadcast_stress_case_312() {
        let t = Tensor::full(vec![1, 2], 312.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 312.0);
        assert_eq!(b.get(5), 312.0);
    }

    #[test]
    fn test_broadcast_stress_case_313() {
        let t = Tensor::full(vec![1, 2], 313.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 313.0);
        assert_eq!(b.get(5), 313.0);
    }

    #[test]
    fn test_broadcast_stress_case_314() {
        let t = Tensor::full(vec![1, 2], 314.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 314.0);
        assert_eq!(b.get(5), 314.0);
    }

    #[test]
    fn test_broadcast_stress_case_315() {
        let t = Tensor::full(vec![1, 2], 315.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 315.0);
        assert_eq!(b.get(5), 315.0);
    }

    #[test]
    fn test_broadcast_stress_case_316() {
        let t = Tensor::full(vec![1, 2], 316.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 316.0);
        assert_eq!(b.get(5), 316.0);
    }

    #[test]
    fn test_broadcast_stress_case_317() {
        let t = Tensor::full(vec![1, 2], 317.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 317.0);
        assert_eq!(b.get(5), 317.0);
    }

    #[test]
    fn test_broadcast_stress_case_318() {
        let t = Tensor::full(vec![1, 2], 318.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 318.0);
        assert_eq!(b.get(5), 318.0);
    }

    #[test]
    fn test_broadcast_stress_case_319() {
        let t = Tensor::full(vec![1, 2], 319.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 319.0);
        assert_eq!(b.get(5), 319.0);
    }

    #[test]
    fn test_broadcast_stress_case_320() {
        let t = Tensor::full(vec![1, 2], 320.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 320.0);
        assert_eq!(b.get(5), 320.0);
    }

    #[test]
    fn test_broadcast_stress_case_321() {
        let t = Tensor::full(vec![1, 2], 321.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 321.0);
        assert_eq!(b.get(5), 321.0);
    }

    #[test]
    fn test_broadcast_stress_case_322() {
        let t = Tensor::full(vec![1, 2], 322.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 322.0);
        assert_eq!(b.get(5), 322.0);
    }

    #[test]
    fn test_broadcast_stress_case_323() {
        let t = Tensor::full(vec![1, 2], 323.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 323.0);
        assert_eq!(b.get(5), 323.0);
    }

    #[test]
    fn test_broadcast_stress_case_324() {
        let t = Tensor::full(vec![1, 2], 324.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 324.0);
        assert_eq!(b.get(5), 324.0);
    }

    #[test]
    fn test_broadcast_stress_case_325() {
        let t = Tensor::full(vec![1, 2], 325.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 325.0);
        assert_eq!(b.get(5), 325.0);
    }

    #[test]
    fn test_broadcast_stress_case_326() {
        let t = Tensor::full(vec![1, 2], 326.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 326.0);
        assert_eq!(b.get(5), 326.0);
    }

    #[test]
    fn test_broadcast_stress_case_327() {
        let t = Tensor::full(vec![1, 2], 327.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 327.0);
        assert_eq!(b.get(5), 327.0);
    }

    #[test]
    fn test_broadcast_stress_case_328() {
        let t = Tensor::full(vec![1, 2], 328.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 328.0);
        assert_eq!(b.get(5), 328.0);
    }

    #[test]
    fn test_broadcast_stress_case_329() {
        let t = Tensor::full(vec![1, 2], 329.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 329.0);
        assert_eq!(b.get(5), 329.0);
    }

    #[test]
    fn test_broadcast_stress_case_330() {
        let t = Tensor::full(vec![1, 2], 330.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 330.0);
        assert_eq!(b.get(5), 330.0);
    }

    #[test]
    fn test_broadcast_stress_case_331() {
        let t = Tensor::full(vec![1, 2], 331.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 331.0);
        assert_eq!(b.get(5), 331.0);
    }

    #[test]
    fn test_broadcast_stress_case_332() {
        let t = Tensor::full(vec![1, 2], 332.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 332.0);
        assert_eq!(b.get(5), 332.0);
    }

    #[test]
    fn test_broadcast_stress_case_333() {
        let t = Tensor::full(vec![1, 2], 333.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 333.0);
        assert_eq!(b.get(5), 333.0);
    }

    #[test]
    fn test_broadcast_stress_case_334() {
        let t = Tensor::full(vec![1, 2], 334.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 334.0);
        assert_eq!(b.get(5), 334.0);
    }

    #[test]
    fn test_broadcast_stress_case_335() {
        let t = Tensor::full(vec![1, 2], 335.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 335.0);
        assert_eq!(b.get(5), 335.0);
    }

    #[test]
    fn test_broadcast_stress_case_336() {
        let t = Tensor::full(vec![1, 2], 336.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 336.0);
        assert_eq!(b.get(5), 336.0);
    }

    #[test]
    fn test_broadcast_stress_case_337() {
        let t = Tensor::full(vec![1, 2], 337.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 337.0);
        assert_eq!(b.get(5), 337.0);
    }

    #[test]
    fn test_broadcast_stress_case_338() {
        let t = Tensor::full(vec![1, 2], 338.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 338.0);
        assert_eq!(b.get(5), 338.0);
    }

    #[test]
    fn test_broadcast_stress_case_339() {
        let t = Tensor::full(vec![1, 2], 339.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 339.0);
        assert_eq!(b.get(5), 339.0);
    }

    #[test]
    fn test_broadcast_stress_case_340() {
        let t = Tensor::full(vec![1, 2], 340.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 340.0);
        assert_eq!(b.get(5), 340.0);
    }

    #[test]
    fn test_broadcast_stress_case_341() {
        let t = Tensor::full(vec![1, 2], 341.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 341.0);
        assert_eq!(b.get(5), 341.0);
    }

    #[test]
    fn test_broadcast_stress_case_342() {
        let t = Tensor::full(vec![1, 2], 342.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 342.0);
        assert_eq!(b.get(5), 342.0);
    }

    #[test]
    fn test_broadcast_stress_case_343() {
        let t = Tensor::full(vec![1, 2], 343.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 343.0);
        assert_eq!(b.get(5), 343.0);
    }

    #[test]
    fn test_broadcast_stress_case_344() {
        let t = Tensor::full(vec![1, 2], 344.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 344.0);
        assert_eq!(b.get(5), 344.0);
    }

    #[test]
    fn test_broadcast_stress_case_345() {
        let t = Tensor::full(vec![1, 2], 345.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 345.0);
        assert_eq!(b.get(5), 345.0);
    }

    #[test]
    fn test_broadcast_stress_case_346() {
        let t = Tensor::full(vec![1, 2], 346.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 346.0);
        assert_eq!(b.get(5), 346.0);
    }

    #[test]
    fn test_broadcast_stress_case_347() {
        let t = Tensor::full(vec![1, 2], 347.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 347.0);
        assert_eq!(b.get(5), 347.0);
    }

    #[test]
    fn test_broadcast_stress_case_348() {
        let t = Tensor::full(vec![1, 2], 348.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 348.0);
        assert_eq!(b.get(5), 348.0);
    }

    #[test]
    fn test_broadcast_stress_case_349() {
        let t = Tensor::full(vec![1, 2], 349.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 349.0);
        assert_eq!(b.get(5), 349.0);
    }

    #[test]
    fn test_broadcast_stress_case_350() {
        let t = Tensor::full(vec![1, 2], 350.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 350.0);
        assert_eq!(b.get(5), 350.0);
    }

    #[test]
    fn test_broadcast_stress_case_351() {
        let t = Tensor::full(vec![1, 2], 351.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 351.0);
        assert_eq!(b.get(5), 351.0);
    }

    #[test]
    fn test_broadcast_stress_case_352() {
        let t = Tensor::full(vec![1, 2], 352.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 352.0);
        assert_eq!(b.get(5), 352.0);
    }

    #[test]
    fn test_broadcast_stress_case_353() {
        let t = Tensor::full(vec![1, 2], 353.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 353.0);
        assert_eq!(b.get(5), 353.0);
    }

    #[test]
    fn test_broadcast_stress_case_354() {
        let t = Tensor::full(vec![1, 2], 354.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 354.0);
        assert_eq!(b.get(5), 354.0);
    }

    #[test]
    fn test_broadcast_stress_case_355() {
        let t = Tensor::full(vec![1, 2], 355.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 355.0);
        assert_eq!(b.get(5), 355.0);
    }

    #[test]
    fn test_broadcast_stress_case_356() {
        let t = Tensor::full(vec![1, 2], 356.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 356.0);
        assert_eq!(b.get(5), 356.0);
    }

    #[test]
    fn test_broadcast_stress_case_357() {
        let t = Tensor::full(vec![1, 2], 357.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 357.0);
        assert_eq!(b.get(5), 357.0);
    }

    #[test]
    fn test_broadcast_stress_case_358() {
        let t = Tensor::full(vec![1, 2], 358.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 358.0);
        assert_eq!(b.get(5), 358.0);
    }

    #[test]
    fn test_broadcast_stress_case_359() {
        let t = Tensor::full(vec![1, 2], 359.0);
        let b = broadcast_to(&t, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.get(0), 359.0);
        assert_eq!(b.get(5), 359.0);
    }
}
