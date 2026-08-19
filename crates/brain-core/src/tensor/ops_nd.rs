//! N-Dimensional tensor manipulation (Concat, Stack, Tile, Repeat, Roll, Rot90).
//!
//! This module provides multi-dimensional joining, splitting, tiling, and spatial rotation operations.

use crate::tensor::Tensor;

/// Concatenates a sequence of tensors along a specified dimension.
pub fn cat(tensors: &[&Tensor], dim: usize) -> Tensor {
    assert!(!tensors.is_empty(), "cat: empty tensor list");
    let rank = tensors[0].ndim();
    assert!(dim < rank);

    let mut out_shape = tensors[0].shape().to_vec();
    let mut total_dim = 0;
    for t in tensors {
        assert_eq!(t.ndim(), rank, "cat: all tensors must have same rank");
        for d in 0..rank {
            if d != dim {
                assert_eq!(t.shape()[d], out_shape[d], "cat: non-concatenating dimension sizes must match");
            }
        }
        total_dim += t.shape()[dim];
    }
    out_shape[dim] = total_dim;

    let numel: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(numel);

    let outer_size: usize = out_shape[..dim].iter().product();
    let inner_size: usize = out_shape[dim + 1..].iter().product();

    for o in 0..outer_size {
        for t in tensors {
            let d_len = t.shape()[dim];
            let block_size = d_len * inner_size;
            let src_offset = o * block_size;
            out.extend_from_slice(&t.data()[src_offset..src_offset + block_size]);
        }
    }

    Tensor::new(out, out_shape)
}

/// Stacks a sequence of tensors along a new dimension.
pub fn stack(tensors: &[&Tensor], dim: usize) -> Tensor {
    assert!(!tensors.is_empty());
    let unsqueezed: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();
    let refs: Vec<&Tensor> = unsqueezed.iter().collect();
    cat(&refs, dim)
}

/// Stacks tensors horizontally (along axis 1, or axis 0 for 1D).
pub fn hstack(tensors: &[&Tensor]) -> Tensor {
    if tensors[0].ndim() == 1 {
        cat(tensors, 0)
    } else {
        cat(tensors, 1)
    }
}

/// Stacks tensors vertically (along axis 0).
pub fn vstack(tensors: &[&Tensor]) -> Tensor {
    cat(tensors, 0)
}

/// Rolls tensor elements along a given dimension by `shift`.
pub fn roll(input: &Tensor, shift: isize, dim: usize) -> Tensor {
    assert!(dim < input.ndim());
    let dim_len = input.shape()[dim];
    let actual_shift = shift.rem_euclid(dim_len as isize) as usize;

    let mut out = Vec::with_capacity(input.numel());
    let rank = input.ndim();
    let mut coords = vec![0usize; rank];

    for _ in 0..input.numel() {
        let mut src_coords = coords.clone();
        src_coords[dim] = (coords[dim] + dim_len - actual_shift) % dim_len;
        out.push(input.get_index(&src_coords));

        for d in (0..rank).rev() {
            coords[d] += 1;
            if coords[d] < input.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Tensor::new(out, input.shape().to_vec())
}

/// Tiles the tensor along each dimension by the given repeat counts (torch `repeat` semantics).
pub fn repeat(input: &Tensor, repeats: &[usize]) -> Tensor {
    assert_eq!(repeats.len(), input.ndim(), "repeat counts must match tensor rank");
    let out_shape: Vec<usize> = input.shape().iter().zip(repeats).map(|(&s, &r)| s * r).collect();
    let rank = input.ndim();
    let mut out_data = vec![0.0; out_shape.iter().product()];
    let mut coords = vec![0usize; rank];

    for flat in 0..out_data.len() {
        let mut rem = flat;
        for d in (0..rank).rev() {
            coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        let src_coords: Vec<usize> = coords.iter().zip(input.shape()).map(|(&c, &s)| c % s).collect();
        out_data[flat] = input.get_index(&src_coords);
    }

    Tensor::new(out_data, out_shape)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cat_and_stack() {
        let t1 = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let t2 = Tensor::from_slice(&[3.0, 4.0], vec![2]);
        let c = cat(&[&t1, &t2], 0);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);

        let s = stack(&[&t1, &t2], 0);
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.get_2d(0, 0), 1.0);
        assert_eq!(s.get_2d(1, 0), 3.0);
    }

    #[test]
    fn test_roll() {
        let t = Tensor::arange(0.0, 4.0, 1.0);
        let r = roll(&t, 1, 0);
        assert_eq!(r.data(), &[3.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let r = repeat(&t, &[2]);
        assert_eq!(r.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(r.shape(), &[6]);

        let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = repeat(&m, &[2, 3]);
        assert_eq!(r.shape(), &[4, 6]);
        assert_eq!(r.get_2d(0, 0), 1.0);
        assert_eq!(r.get_2d(0, 3), 2.0);
        assert_eq!(r.get_2d(2, 0), 1.0);
        assert_eq!(r.get_2d(2, 3), 2.0);
        assert_eq!(r.get_2d(3, 5), 4.0);
    }

    #[test]
    fn test_cat_and_stack_edge_cases() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let b = Tensor::from_slice(&[3.0, 4.0], vec![1, 2]);
        
        let cat0 = cat(&[&a, &b], 0);
        assert_eq!(cat0.shape(), &[2, 2]);
        assert_eq!(cat0.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        let cat1 = cat(&[&a, &b], 1);
        assert_eq!(cat1.shape(), &[1, 4]);
        assert_eq!(cat1.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        let st = stack(&[&a, &b], 0);
        assert_eq!(st.shape(), &[2, 1, 2]);
    }
}
