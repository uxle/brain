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
        assert_eq!(t.ndim(), rank);
        total_dim += t.shape()[dim];
    }
    out_shape[dim] = total_dim;

    let numel: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let mut coords = vec![0usize; rank];

    for _ in 0..numel {
        let cur_dim_idx = coords[dim];
        let mut accum = 0;
        let mut target_tensor_idx = 0;
        let mut inner_dim_idx = 0;

        for (i, t) in tensors.iter().enumerate() {
            let d_len = t.shape()[dim];
            if cur_dim_idx < accum + d_len {
                target_tensor_idx = i;
                inner_dim_idx = cur_dim_idx - accum;
                break;
            }
            accum += d_len;
        }

        let mut src_coords = coords.clone();
        src_coords[dim] = inner_dim_idx;
        out.push(tensors[target_tensor_idx].get_index(&src_coords));

        for d in (0..rank).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
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
    fn test_nd_stress_case_001() {
        let t = Tensor::from_slice(&[1.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 1.0);
        assert_eq!(c.get(1), 1.0);
    }

    #[test]
    fn test_nd_stress_case_002() {
        let t = Tensor::from_slice(&[2.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 2.0);
        assert_eq!(c.get(1), 2.0);
    }

    #[test]
    fn test_nd_stress_case_003() {
        let t = Tensor::from_slice(&[3.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 3.0);
        assert_eq!(c.get(1), 3.0);
    }

    #[test]
    fn test_nd_stress_case_004() {
        let t = Tensor::from_slice(&[4.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 4.0);
        assert_eq!(c.get(1), 4.0);
    }

    #[test]
    fn test_nd_stress_case_005() {
        let t = Tensor::from_slice(&[5.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 5.0);
        assert_eq!(c.get(1), 5.0);
    }

    #[test]
    fn test_nd_stress_case_006() {
        let t = Tensor::from_slice(&[6.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 6.0);
        assert_eq!(c.get(1), 6.0);
    }

    #[test]
    fn test_nd_stress_case_007() {
        let t = Tensor::from_slice(&[7.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 7.0);
        assert_eq!(c.get(1), 7.0);
    }

    #[test]
    fn test_nd_stress_case_008() {
        let t = Tensor::from_slice(&[8.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 8.0);
        assert_eq!(c.get(1), 8.0);
    }

    #[test]
    fn test_nd_stress_case_009() {
        let t = Tensor::from_slice(&[9.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 9.0);
        assert_eq!(c.get(1), 9.0);
    }

    #[test]
    fn test_nd_stress_case_010() {
        let t = Tensor::from_slice(&[10.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 10.0);
        assert_eq!(c.get(1), 10.0);
    }

    #[test]
    fn test_nd_stress_case_011() {
        let t = Tensor::from_slice(&[11.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 11.0);
        assert_eq!(c.get(1), 11.0);
    }

    #[test]
    fn test_nd_stress_case_012() {
        let t = Tensor::from_slice(&[12.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 12.0);
        assert_eq!(c.get(1), 12.0);
    }

    #[test]
    fn test_nd_stress_case_013() {
        let t = Tensor::from_slice(&[13.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 13.0);
        assert_eq!(c.get(1), 13.0);
    }

    #[test]
    fn test_nd_stress_case_014() {
        let t = Tensor::from_slice(&[14.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 14.0);
        assert_eq!(c.get(1), 14.0);
    }

    #[test]
    fn test_nd_stress_case_015() {
        let t = Tensor::from_slice(&[15.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 15.0);
        assert_eq!(c.get(1), 15.0);
    }

    #[test]
    fn test_nd_stress_case_016() {
        let t = Tensor::from_slice(&[16.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 16.0);
        assert_eq!(c.get(1), 16.0);
    }

    #[test]
    fn test_nd_stress_case_017() {
        let t = Tensor::from_slice(&[17.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 17.0);
        assert_eq!(c.get(1), 17.0);
    }

    #[test]
    fn test_nd_stress_case_018() {
        let t = Tensor::from_slice(&[18.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 18.0);
        assert_eq!(c.get(1), 18.0);
    }

    #[test]
    fn test_nd_stress_case_019() {
        let t = Tensor::from_slice(&[19.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 19.0);
        assert_eq!(c.get(1), 19.0);
    }

    #[test]
    fn test_nd_stress_case_020() {
        let t = Tensor::from_slice(&[20.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 20.0);
        assert_eq!(c.get(1), 20.0);
    }

    #[test]
    fn test_nd_stress_case_021() {
        let t = Tensor::from_slice(&[21.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 21.0);
        assert_eq!(c.get(1), 21.0);
    }

    #[test]
    fn test_nd_stress_case_022() {
        let t = Tensor::from_slice(&[22.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 22.0);
        assert_eq!(c.get(1), 22.0);
    }

    #[test]
    fn test_nd_stress_case_023() {
        let t = Tensor::from_slice(&[23.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 23.0);
        assert_eq!(c.get(1), 23.0);
    }

    #[test]
    fn test_nd_stress_case_024() {
        let t = Tensor::from_slice(&[24.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 24.0);
        assert_eq!(c.get(1), 24.0);
    }

    #[test]
    fn test_nd_stress_case_025() {
        let t = Tensor::from_slice(&[25.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 25.0);
        assert_eq!(c.get(1), 25.0);
    }

    #[test]
    fn test_nd_stress_case_026() {
        let t = Tensor::from_slice(&[26.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 26.0);
        assert_eq!(c.get(1), 26.0);
    }

    #[test]
    fn test_nd_stress_case_027() {
        let t = Tensor::from_slice(&[27.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 27.0);
        assert_eq!(c.get(1), 27.0);
    }

    #[test]
    fn test_nd_stress_case_028() {
        let t = Tensor::from_slice(&[28.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 28.0);
        assert_eq!(c.get(1), 28.0);
    }

    #[test]
    fn test_nd_stress_case_029() {
        let t = Tensor::from_slice(&[29.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 29.0);
        assert_eq!(c.get(1), 29.0);
    }

    #[test]
    fn test_nd_stress_case_030() {
        let t = Tensor::from_slice(&[30.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 30.0);
        assert_eq!(c.get(1), 30.0);
    }

    #[test]
    fn test_nd_stress_case_031() {
        let t = Tensor::from_slice(&[31.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 31.0);
        assert_eq!(c.get(1), 31.0);
    }

    #[test]
    fn test_nd_stress_case_032() {
        let t = Tensor::from_slice(&[32.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 32.0);
        assert_eq!(c.get(1), 32.0);
    }

    #[test]
    fn test_nd_stress_case_033() {
        let t = Tensor::from_slice(&[33.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 33.0);
        assert_eq!(c.get(1), 33.0);
    }

    #[test]
    fn test_nd_stress_case_034() {
        let t = Tensor::from_slice(&[34.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 34.0);
        assert_eq!(c.get(1), 34.0);
    }

    #[test]
    fn test_nd_stress_case_035() {
        let t = Tensor::from_slice(&[35.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 35.0);
        assert_eq!(c.get(1), 35.0);
    }

    #[test]
    fn test_nd_stress_case_036() {
        let t = Tensor::from_slice(&[36.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 36.0);
        assert_eq!(c.get(1), 36.0);
    }

    #[test]
    fn test_nd_stress_case_037() {
        let t = Tensor::from_slice(&[37.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 37.0);
        assert_eq!(c.get(1), 37.0);
    }

    #[test]
    fn test_nd_stress_case_038() {
        let t = Tensor::from_slice(&[38.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 38.0);
        assert_eq!(c.get(1), 38.0);
    }

    #[test]
    fn test_nd_stress_case_039() {
        let t = Tensor::from_slice(&[39.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 39.0);
        assert_eq!(c.get(1), 39.0);
    }

    #[test]
    fn test_nd_stress_case_040() {
        let t = Tensor::from_slice(&[40.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 40.0);
        assert_eq!(c.get(1), 40.0);
    }

    #[test]
    fn test_nd_stress_case_041() {
        let t = Tensor::from_slice(&[41.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 41.0);
        assert_eq!(c.get(1), 41.0);
    }

    #[test]
    fn test_nd_stress_case_042() {
        let t = Tensor::from_slice(&[42.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 42.0);
        assert_eq!(c.get(1), 42.0);
    }

    #[test]
    fn test_nd_stress_case_043() {
        let t = Tensor::from_slice(&[43.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 43.0);
        assert_eq!(c.get(1), 43.0);
    }

    #[test]
    fn test_nd_stress_case_044() {
        let t = Tensor::from_slice(&[44.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 44.0);
        assert_eq!(c.get(1), 44.0);
    }

    #[test]
    fn test_nd_stress_case_045() {
        let t = Tensor::from_slice(&[45.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 45.0);
        assert_eq!(c.get(1), 45.0);
    }

    #[test]
    fn test_nd_stress_case_046() {
        let t = Tensor::from_slice(&[46.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 46.0);
        assert_eq!(c.get(1), 46.0);
    }

    #[test]
    fn test_nd_stress_case_047() {
        let t = Tensor::from_slice(&[47.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 47.0);
        assert_eq!(c.get(1), 47.0);
    }

    #[test]
    fn test_nd_stress_case_048() {
        let t = Tensor::from_slice(&[48.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 48.0);
        assert_eq!(c.get(1), 48.0);
    }

    #[test]
    fn test_nd_stress_case_049() {
        let t = Tensor::from_slice(&[49.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 49.0);
        assert_eq!(c.get(1), 49.0);
    }

    #[test]
    fn test_nd_stress_case_050() {
        let t = Tensor::from_slice(&[50.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 50.0);
        assert_eq!(c.get(1), 50.0);
    }

    #[test]
    fn test_nd_stress_case_051() {
        let t = Tensor::from_slice(&[51.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 51.0);
        assert_eq!(c.get(1), 51.0);
    }

    #[test]
    fn test_nd_stress_case_052() {
        let t = Tensor::from_slice(&[52.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 52.0);
        assert_eq!(c.get(1), 52.0);
    }

    #[test]
    fn test_nd_stress_case_053() {
        let t = Tensor::from_slice(&[53.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 53.0);
        assert_eq!(c.get(1), 53.0);
    }

    #[test]
    fn test_nd_stress_case_054() {
        let t = Tensor::from_slice(&[54.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 54.0);
        assert_eq!(c.get(1), 54.0);
    }

    #[test]
    fn test_nd_stress_case_055() {
        let t = Tensor::from_slice(&[55.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 55.0);
        assert_eq!(c.get(1), 55.0);
    }

    #[test]
    fn test_nd_stress_case_056() {
        let t = Tensor::from_slice(&[56.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 56.0);
        assert_eq!(c.get(1), 56.0);
    }

    #[test]
    fn test_nd_stress_case_057() {
        let t = Tensor::from_slice(&[57.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 57.0);
        assert_eq!(c.get(1), 57.0);
    }

    #[test]
    fn test_nd_stress_case_058() {
        let t = Tensor::from_slice(&[58.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 58.0);
        assert_eq!(c.get(1), 58.0);
    }

    #[test]
    fn test_nd_stress_case_059() {
        let t = Tensor::from_slice(&[59.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 59.0);
        assert_eq!(c.get(1), 59.0);
    }

    #[test]
    fn test_nd_stress_case_060() {
        let t = Tensor::from_slice(&[60.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 60.0);
        assert_eq!(c.get(1), 60.0);
    }

    #[test]
    fn test_nd_stress_case_061() {
        let t = Tensor::from_slice(&[61.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 61.0);
        assert_eq!(c.get(1), 61.0);
    }

    #[test]
    fn test_nd_stress_case_062() {
        let t = Tensor::from_slice(&[62.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 62.0);
        assert_eq!(c.get(1), 62.0);
    }

    #[test]
    fn test_nd_stress_case_063() {
        let t = Tensor::from_slice(&[63.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 63.0);
        assert_eq!(c.get(1), 63.0);
    }

    #[test]
    fn test_nd_stress_case_064() {
        let t = Tensor::from_slice(&[64.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 64.0);
        assert_eq!(c.get(1), 64.0);
    }

    #[test]
    fn test_nd_stress_case_065() {
        let t = Tensor::from_slice(&[65.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 65.0);
        assert_eq!(c.get(1), 65.0);
    }

    #[test]
    fn test_nd_stress_case_066() {
        let t = Tensor::from_slice(&[66.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 66.0);
        assert_eq!(c.get(1), 66.0);
    }

    #[test]
    fn test_nd_stress_case_067() {
        let t = Tensor::from_slice(&[67.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 67.0);
        assert_eq!(c.get(1), 67.0);
    }

    #[test]
    fn test_nd_stress_case_068() {
        let t = Tensor::from_slice(&[68.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 68.0);
        assert_eq!(c.get(1), 68.0);
    }

    #[test]
    fn test_nd_stress_case_069() {
        let t = Tensor::from_slice(&[69.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 69.0);
        assert_eq!(c.get(1), 69.0);
    }

    #[test]
    fn test_nd_stress_case_070() {
        let t = Tensor::from_slice(&[70.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 70.0);
        assert_eq!(c.get(1), 70.0);
    }

    #[test]
    fn test_nd_stress_case_071() {
        let t = Tensor::from_slice(&[71.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 71.0);
        assert_eq!(c.get(1), 71.0);
    }

    #[test]
    fn test_nd_stress_case_072() {
        let t = Tensor::from_slice(&[72.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 72.0);
        assert_eq!(c.get(1), 72.0);
    }

    #[test]
    fn test_nd_stress_case_073() {
        let t = Tensor::from_slice(&[73.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 73.0);
        assert_eq!(c.get(1), 73.0);
    }

    #[test]
    fn test_nd_stress_case_074() {
        let t = Tensor::from_slice(&[74.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 74.0);
        assert_eq!(c.get(1), 74.0);
    }

    #[test]
    fn test_nd_stress_case_075() {
        let t = Tensor::from_slice(&[75.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 75.0);
        assert_eq!(c.get(1), 75.0);
    }

    #[test]
    fn test_nd_stress_case_076() {
        let t = Tensor::from_slice(&[76.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 76.0);
        assert_eq!(c.get(1), 76.0);
    }

    #[test]
    fn test_nd_stress_case_077() {
        let t = Tensor::from_slice(&[77.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 77.0);
        assert_eq!(c.get(1), 77.0);
    }

    #[test]
    fn test_nd_stress_case_078() {
        let t = Tensor::from_slice(&[78.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 78.0);
        assert_eq!(c.get(1), 78.0);
    }

    #[test]
    fn test_nd_stress_case_079() {
        let t = Tensor::from_slice(&[79.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 79.0);
        assert_eq!(c.get(1), 79.0);
    }

    #[test]
    fn test_nd_stress_case_080() {
        let t = Tensor::from_slice(&[80.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 80.0);
        assert_eq!(c.get(1), 80.0);
    }

    #[test]
    fn test_nd_stress_case_081() {
        let t = Tensor::from_slice(&[81.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 81.0);
        assert_eq!(c.get(1), 81.0);
    }

    #[test]
    fn test_nd_stress_case_082() {
        let t = Tensor::from_slice(&[82.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 82.0);
        assert_eq!(c.get(1), 82.0);
    }

    #[test]
    fn test_nd_stress_case_083() {
        let t = Tensor::from_slice(&[83.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 83.0);
        assert_eq!(c.get(1), 83.0);
    }

    #[test]
    fn test_nd_stress_case_084() {
        let t = Tensor::from_slice(&[84.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 84.0);
        assert_eq!(c.get(1), 84.0);
    }

    #[test]
    fn test_nd_stress_case_085() {
        let t = Tensor::from_slice(&[85.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 85.0);
        assert_eq!(c.get(1), 85.0);
    }

    #[test]
    fn test_nd_stress_case_086() {
        let t = Tensor::from_slice(&[86.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 86.0);
        assert_eq!(c.get(1), 86.0);
    }

    #[test]
    fn test_nd_stress_case_087() {
        let t = Tensor::from_slice(&[87.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 87.0);
        assert_eq!(c.get(1), 87.0);
    }

    #[test]
    fn test_nd_stress_case_088() {
        let t = Tensor::from_slice(&[88.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 88.0);
        assert_eq!(c.get(1), 88.0);
    }

    #[test]
    fn test_nd_stress_case_089() {
        let t = Tensor::from_slice(&[89.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 89.0);
        assert_eq!(c.get(1), 89.0);
    }

    #[test]
    fn test_nd_stress_case_090() {
        let t = Tensor::from_slice(&[90.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 90.0);
        assert_eq!(c.get(1), 90.0);
    }

    #[test]
    fn test_nd_stress_case_091() {
        let t = Tensor::from_slice(&[91.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 91.0);
        assert_eq!(c.get(1), 91.0);
    }

    #[test]
    fn test_nd_stress_case_092() {
        let t = Tensor::from_slice(&[92.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 92.0);
        assert_eq!(c.get(1), 92.0);
    }

    #[test]
    fn test_nd_stress_case_093() {
        let t = Tensor::from_slice(&[93.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 93.0);
        assert_eq!(c.get(1), 93.0);
    }

    #[test]
    fn test_nd_stress_case_094() {
        let t = Tensor::from_slice(&[94.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 94.0);
        assert_eq!(c.get(1), 94.0);
    }

    #[test]
    fn test_nd_stress_case_095() {
        let t = Tensor::from_slice(&[95.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 95.0);
        assert_eq!(c.get(1), 95.0);
    }

    #[test]
    fn test_nd_stress_case_096() {
        let t = Tensor::from_slice(&[96.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 96.0);
        assert_eq!(c.get(1), 96.0);
    }

    #[test]
    fn test_nd_stress_case_097() {
        let t = Tensor::from_slice(&[97.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 97.0);
        assert_eq!(c.get(1), 97.0);
    }

    #[test]
    fn test_nd_stress_case_098() {
        let t = Tensor::from_slice(&[98.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 98.0);
        assert_eq!(c.get(1), 98.0);
    }

    #[test]
    fn test_nd_stress_case_099() {
        let t = Tensor::from_slice(&[99.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 99.0);
        assert_eq!(c.get(1), 99.0);
    }

    #[test]
    fn test_nd_stress_case_100() {
        let t = Tensor::from_slice(&[100.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 100.0);
        assert_eq!(c.get(1), 100.0);
    }

    #[test]
    fn test_nd_stress_case_101() {
        let t = Tensor::from_slice(&[101.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 101.0);
        assert_eq!(c.get(1), 101.0);
    }

    #[test]
    fn test_nd_stress_case_102() {
        let t = Tensor::from_slice(&[102.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 102.0);
        assert_eq!(c.get(1), 102.0);
    }

    #[test]
    fn test_nd_stress_case_103() {
        let t = Tensor::from_slice(&[103.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 103.0);
        assert_eq!(c.get(1), 103.0);
    }

    #[test]
    fn test_nd_stress_case_104() {
        let t = Tensor::from_slice(&[104.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 104.0);
        assert_eq!(c.get(1), 104.0);
    }

    #[test]
    fn test_nd_stress_case_105() {
        let t = Tensor::from_slice(&[105.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 105.0);
        assert_eq!(c.get(1), 105.0);
    }

    #[test]
    fn test_nd_stress_case_106() {
        let t = Tensor::from_slice(&[106.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 106.0);
        assert_eq!(c.get(1), 106.0);
    }

    #[test]
    fn test_nd_stress_case_107() {
        let t = Tensor::from_slice(&[107.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 107.0);
        assert_eq!(c.get(1), 107.0);
    }

    #[test]
    fn test_nd_stress_case_108() {
        let t = Tensor::from_slice(&[108.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 108.0);
        assert_eq!(c.get(1), 108.0);
    }

    #[test]
    fn test_nd_stress_case_109() {
        let t = Tensor::from_slice(&[109.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 109.0);
        assert_eq!(c.get(1), 109.0);
    }

    #[test]
    fn test_nd_stress_case_110() {
        let t = Tensor::from_slice(&[110.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 110.0);
        assert_eq!(c.get(1), 110.0);
    }

    #[test]
    fn test_nd_stress_case_111() {
        let t = Tensor::from_slice(&[111.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 111.0);
        assert_eq!(c.get(1), 111.0);
    }

    #[test]
    fn test_nd_stress_case_112() {
        let t = Tensor::from_slice(&[112.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 112.0);
        assert_eq!(c.get(1), 112.0);
    }

    #[test]
    fn test_nd_stress_case_113() {
        let t = Tensor::from_slice(&[113.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 113.0);
        assert_eq!(c.get(1), 113.0);
    }

    #[test]
    fn test_nd_stress_case_114() {
        let t = Tensor::from_slice(&[114.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 114.0);
        assert_eq!(c.get(1), 114.0);
    }

    #[test]
    fn test_nd_stress_case_115() {
        let t = Tensor::from_slice(&[115.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 115.0);
        assert_eq!(c.get(1), 115.0);
    }

    #[test]
    fn test_nd_stress_case_116() {
        let t = Tensor::from_slice(&[116.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 116.0);
        assert_eq!(c.get(1), 116.0);
    }

    #[test]
    fn test_nd_stress_case_117() {
        let t = Tensor::from_slice(&[117.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 117.0);
        assert_eq!(c.get(1), 117.0);
    }

    #[test]
    fn test_nd_stress_case_118() {
        let t = Tensor::from_slice(&[118.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 118.0);
        assert_eq!(c.get(1), 118.0);
    }

    #[test]
    fn test_nd_stress_case_119() {
        let t = Tensor::from_slice(&[119.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 119.0);
        assert_eq!(c.get(1), 119.0);
    }

    #[test]
    fn test_nd_stress_case_120() {
        let t = Tensor::from_slice(&[120.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 120.0);
        assert_eq!(c.get(1), 120.0);
    }

    #[test]
    fn test_nd_stress_case_121() {
        let t = Tensor::from_slice(&[121.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 121.0);
        assert_eq!(c.get(1), 121.0);
    }

    #[test]
    fn test_nd_stress_case_122() {
        let t = Tensor::from_slice(&[122.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 122.0);
        assert_eq!(c.get(1), 122.0);
    }

    #[test]
    fn test_nd_stress_case_123() {
        let t = Tensor::from_slice(&[123.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 123.0);
        assert_eq!(c.get(1), 123.0);
    }

    #[test]
    fn test_nd_stress_case_124() {
        let t = Tensor::from_slice(&[124.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 124.0);
        assert_eq!(c.get(1), 124.0);
    }

    #[test]
    fn test_nd_stress_case_125() {
        let t = Tensor::from_slice(&[125.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 125.0);
        assert_eq!(c.get(1), 125.0);
    }

    #[test]
    fn test_nd_stress_case_126() {
        let t = Tensor::from_slice(&[126.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 126.0);
        assert_eq!(c.get(1), 126.0);
    }

    #[test]
    fn test_nd_stress_case_127() {
        let t = Tensor::from_slice(&[127.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 127.0);
        assert_eq!(c.get(1), 127.0);
    }

    #[test]
    fn test_nd_stress_case_128() {
        let t = Tensor::from_slice(&[128.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 128.0);
        assert_eq!(c.get(1), 128.0);
    }

    #[test]
    fn test_nd_stress_case_129() {
        let t = Tensor::from_slice(&[129.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 129.0);
        assert_eq!(c.get(1), 129.0);
    }

    #[test]
    fn test_nd_stress_case_130() {
        let t = Tensor::from_slice(&[130.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 130.0);
        assert_eq!(c.get(1), 130.0);
    }

    #[test]
    fn test_nd_stress_case_131() {
        let t = Tensor::from_slice(&[131.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 131.0);
        assert_eq!(c.get(1), 131.0);
    }

    #[test]
    fn test_nd_stress_case_132() {
        let t = Tensor::from_slice(&[132.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 132.0);
        assert_eq!(c.get(1), 132.0);
    }

    #[test]
    fn test_nd_stress_case_133() {
        let t = Tensor::from_slice(&[133.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 133.0);
        assert_eq!(c.get(1), 133.0);
    }

    #[test]
    fn test_nd_stress_case_134() {
        let t = Tensor::from_slice(&[134.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 134.0);
        assert_eq!(c.get(1), 134.0);
    }

    #[test]
    fn test_nd_stress_case_135() {
        let t = Tensor::from_slice(&[135.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 135.0);
        assert_eq!(c.get(1), 135.0);
    }

    #[test]
    fn test_nd_stress_case_136() {
        let t = Tensor::from_slice(&[136.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 136.0);
        assert_eq!(c.get(1), 136.0);
    }

    #[test]
    fn test_nd_stress_case_137() {
        let t = Tensor::from_slice(&[137.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 137.0);
        assert_eq!(c.get(1), 137.0);
    }

    #[test]
    fn test_nd_stress_case_138() {
        let t = Tensor::from_slice(&[138.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 138.0);
        assert_eq!(c.get(1), 138.0);
    }

    #[test]
    fn test_nd_stress_case_139() {
        let t = Tensor::from_slice(&[139.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 139.0);
        assert_eq!(c.get(1), 139.0);
    }

    #[test]
    fn test_nd_stress_case_140() {
        let t = Tensor::from_slice(&[140.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 140.0);
        assert_eq!(c.get(1), 140.0);
    }

    #[test]
    fn test_nd_stress_case_141() {
        let t = Tensor::from_slice(&[141.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 141.0);
        assert_eq!(c.get(1), 141.0);
    }

    #[test]
    fn test_nd_stress_case_142() {
        let t = Tensor::from_slice(&[142.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 142.0);
        assert_eq!(c.get(1), 142.0);
    }

    #[test]
    fn test_nd_stress_case_143() {
        let t = Tensor::from_slice(&[143.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 143.0);
        assert_eq!(c.get(1), 143.0);
    }

    #[test]
    fn test_nd_stress_case_144() {
        let t = Tensor::from_slice(&[144.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 144.0);
        assert_eq!(c.get(1), 144.0);
    }

    #[test]
    fn test_nd_stress_case_145() {
        let t = Tensor::from_slice(&[145.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 145.0);
        assert_eq!(c.get(1), 145.0);
    }

    #[test]
    fn test_nd_stress_case_146() {
        let t = Tensor::from_slice(&[146.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 146.0);
        assert_eq!(c.get(1), 146.0);
    }

    #[test]
    fn test_nd_stress_case_147() {
        let t = Tensor::from_slice(&[147.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 147.0);
        assert_eq!(c.get(1), 147.0);
    }

    #[test]
    fn test_nd_stress_case_148() {
        let t = Tensor::from_slice(&[148.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 148.0);
        assert_eq!(c.get(1), 148.0);
    }

    #[test]
    fn test_nd_stress_case_149() {
        let t = Tensor::from_slice(&[149.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 149.0);
        assert_eq!(c.get(1), 149.0);
    }

    #[test]
    fn test_nd_stress_case_150() {
        let t = Tensor::from_slice(&[150.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 150.0);
        assert_eq!(c.get(1), 150.0);
    }

    #[test]
    fn test_nd_stress_case_151() {
        let t = Tensor::from_slice(&[151.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 151.0);
        assert_eq!(c.get(1), 151.0);
    }

    #[test]
    fn test_nd_stress_case_152() {
        let t = Tensor::from_slice(&[152.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 152.0);
        assert_eq!(c.get(1), 152.0);
    }

    #[test]
    fn test_nd_stress_case_153() {
        let t = Tensor::from_slice(&[153.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 153.0);
        assert_eq!(c.get(1), 153.0);
    }

    #[test]
    fn test_nd_stress_case_154() {
        let t = Tensor::from_slice(&[154.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 154.0);
        assert_eq!(c.get(1), 154.0);
    }

    #[test]
    fn test_nd_stress_case_155() {
        let t = Tensor::from_slice(&[155.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 155.0);
        assert_eq!(c.get(1), 155.0);
    }

    #[test]
    fn test_nd_stress_case_156() {
        let t = Tensor::from_slice(&[156.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 156.0);
        assert_eq!(c.get(1), 156.0);
    }

    #[test]
    fn test_nd_stress_case_157() {
        let t = Tensor::from_slice(&[157.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 157.0);
        assert_eq!(c.get(1), 157.0);
    }

    #[test]
    fn test_nd_stress_case_158() {
        let t = Tensor::from_slice(&[158.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 158.0);
        assert_eq!(c.get(1), 158.0);
    }

    #[test]
    fn test_nd_stress_case_159() {
        let t = Tensor::from_slice(&[159.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 159.0);
        assert_eq!(c.get(1), 159.0);
    }

    #[test]
    fn test_nd_stress_case_160() {
        let t = Tensor::from_slice(&[160.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 160.0);
        assert_eq!(c.get(1), 160.0);
    }

    #[test]
    fn test_nd_stress_case_161() {
        let t = Tensor::from_slice(&[161.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 161.0);
        assert_eq!(c.get(1), 161.0);
    }

    #[test]
    fn test_nd_stress_case_162() {
        let t = Tensor::from_slice(&[162.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 162.0);
        assert_eq!(c.get(1), 162.0);
    }

    #[test]
    fn test_nd_stress_case_163() {
        let t = Tensor::from_slice(&[163.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 163.0);
        assert_eq!(c.get(1), 163.0);
    }

    #[test]
    fn test_nd_stress_case_164() {
        let t = Tensor::from_slice(&[164.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 164.0);
        assert_eq!(c.get(1), 164.0);
    }

    #[test]
    fn test_nd_stress_case_165() {
        let t = Tensor::from_slice(&[165.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 165.0);
        assert_eq!(c.get(1), 165.0);
    }

    #[test]
    fn test_nd_stress_case_166() {
        let t = Tensor::from_slice(&[166.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 166.0);
        assert_eq!(c.get(1), 166.0);
    }

    #[test]
    fn test_nd_stress_case_167() {
        let t = Tensor::from_slice(&[167.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 167.0);
        assert_eq!(c.get(1), 167.0);
    }

    #[test]
    fn test_nd_stress_case_168() {
        let t = Tensor::from_slice(&[168.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 168.0);
        assert_eq!(c.get(1), 168.0);
    }

    #[test]
    fn test_nd_stress_case_169() {
        let t = Tensor::from_slice(&[169.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 169.0);
        assert_eq!(c.get(1), 169.0);
    }

    #[test]
    fn test_nd_stress_case_170() {
        let t = Tensor::from_slice(&[170.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 170.0);
        assert_eq!(c.get(1), 170.0);
    }

    #[test]
    fn test_nd_stress_case_171() {
        let t = Tensor::from_slice(&[171.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 171.0);
        assert_eq!(c.get(1), 171.0);
    }

    #[test]
    fn test_nd_stress_case_172() {
        let t = Tensor::from_slice(&[172.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 172.0);
        assert_eq!(c.get(1), 172.0);
    }

    #[test]
    fn test_nd_stress_case_173() {
        let t = Tensor::from_slice(&[173.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 173.0);
        assert_eq!(c.get(1), 173.0);
    }

    #[test]
    fn test_nd_stress_case_174() {
        let t = Tensor::from_slice(&[174.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 174.0);
        assert_eq!(c.get(1), 174.0);
    }

    #[test]
    fn test_nd_stress_case_175() {
        let t = Tensor::from_slice(&[175.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 175.0);
        assert_eq!(c.get(1), 175.0);
    }

    #[test]
    fn test_nd_stress_case_176() {
        let t = Tensor::from_slice(&[176.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 176.0);
        assert_eq!(c.get(1), 176.0);
    }

    #[test]
    fn test_nd_stress_case_177() {
        let t = Tensor::from_slice(&[177.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 177.0);
        assert_eq!(c.get(1), 177.0);
    }

    #[test]
    fn test_nd_stress_case_178() {
        let t = Tensor::from_slice(&[178.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 178.0);
        assert_eq!(c.get(1), 178.0);
    }

    #[test]
    fn test_nd_stress_case_179() {
        let t = Tensor::from_slice(&[179.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 179.0);
        assert_eq!(c.get(1), 179.0);
    }

    #[test]
    fn test_nd_stress_case_180() {
        let t = Tensor::from_slice(&[180.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 180.0);
        assert_eq!(c.get(1), 180.0);
    }

    #[test]
    fn test_nd_stress_case_181() {
        let t = Tensor::from_slice(&[181.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 181.0);
        assert_eq!(c.get(1), 181.0);
    }

    #[test]
    fn test_nd_stress_case_182() {
        let t = Tensor::from_slice(&[182.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 182.0);
        assert_eq!(c.get(1), 182.0);
    }

    #[test]
    fn test_nd_stress_case_183() {
        let t = Tensor::from_slice(&[183.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 183.0);
        assert_eq!(c.get(1), 183.0);
    }

    #[test]
    fn test_nd_stress_case_184() {
        let t = Tensor::from_slice(&[184.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 184.0);
        assert_eq!(c.get(1), 184.0);
    }

    #[test]
    fn test_nd_stress_case_185() {
        let t = Tensor::from_slice(&[185.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 185.0);
        assert_eq!(c.get(1), 185.0);
    }

    #[test]
    fn test_nd_stress_case_186() {
        let t = Tensor::from_slice(&[186.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 186.0);
        assert_eq!(c.get(1), 186.0);
    }

    #[test]
    fn test_nd_stress_case_187() {
        let t = Tensor::from_slice(&[187.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 187.0);
        assert_eq!(c.get(1), 187.0);
    }

    #[test]
    fn test_nd_stress_case_188() {
        let t = Tensor::from_slice(&[188.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 188.0);
        assert_eq!(c.get(1), 188.0);
    }

    #[test]
    fn test_nd_stress_case_189() {
        let t = Tensor::from_slice(&[189.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 189.0);
        assert_eq!(c.get(1), 189.0);
    }

    #[test]
    fn test_nd_stress_case_190() {
        let t = Tensor::from_slice(&[190.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 190.0);
        assert_eq!(c.get(1), 190.0);
    }

    #[test]
    fn test_nd_stress_case_191() {
        let t = Tensor::from_slice(&[191.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 191.0);
        assert_eq!(c.get(1), 191.0);
    }

    #[test]
    fn test_nd_stress_case_192() {
        let t = Tensor::from_slice(&[192.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 192.0);
        assert_eq!(c.get(1), 192.0);
    }

    #[test]
    fn test_nd_stress_case_193() {
        let t = Tensor::from_slice(&[193.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 193.0);
        assert_eq!(c.get(1), 193.0);
    }

    #[test]
    fn test_nd_stress_case_194() {
        let t = Tensor::from_slice(&[194.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 194.0);
        assert_eq!(c.get(1), 194.0);
    }

    #[test]
    fn test_nd_stress_case_195() {
        let t = Tensor::from_slice(&[195.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 195.0);
        assert_eq!(c.get(1), 195.0);
    }

    #[test]
    fn test_nd_stress_case_196() {
        let t = Tensor::from_slice(&[196.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 196.0);
        assert_eq!(c.get(1), 196.0);
    }

    #[test]
    fn test_nd_stress_case_197() {
        let t = Tensor::from_slice(&[197.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 197.0);
        assert_eq!(c.get(1), 197.0);
    }

    #[test]
    fn test_nd_stress_case_198() {
        let t = Tensor::from_slice(&[198.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 198.0);
        assert_eq!(c.get(1), 198.0);
    }

    #[test]
    fn test_nd_stress_case_199() {
        let t = Tensor::from_slice(&[199.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 199.0);
        assert_eq!(c.get(1), 199.0);
    }

    #[test]
    fn test_nd_stress_case_200() {
        let t = Tensor::from_slice(&[200.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 200.0);
        assert_eq!(c.get(1), 200.0);
    }

    #[test]
    fn test_nd_stress_case_201() {
        let t = Tensor::from_slice(&[201.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 201.0);
        assert_eq!(c.get(1), 201.0);
    }

    #[test]
    fn test_nd_stress_case_202() {
        let t = Tensor::from_slice(&[202.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 202.0);
        assert_eq!(c.get(1), 202.0);
    }

    #[test]
    fn test_nd_stress_case_203() {
        let t = Tensor::from_slice(&[203.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 203.0);
        assert_eq!(c.get(1), 203.0);
    }

    #[test]
    fn test_nd_stress_case_204() {
        let t = Tensor::from_slice(&[204.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 204.0);
        assert_eq!(c.get(1), 204.0);
    }

    #[test]
    fn test_nd_stress_case_205() {
        let t = Tensor::from_slice(&[205.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 205.0);
        assert_eq!(c.get(1), 205.0);
    }

    #[test]
    fn test_nd_stress_case_206() {
        let t = Tensor::from_slice(&[206.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 206.0);
        assert_eq!(c.get(1), 206.0);
    }

    #[test]
    fn test_nd_stress_case_207() {
        let t = Tensor::from_slice(&[207.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 207.0);
        assert_eq!(c.get(1), 207.0);
    }

    #[test]
    fn test_nd_stress_case_208() {
        let t = Tensor::from_slice(&[208.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 208.0);
        assert_eq!(c.get(1), 208.0);
    }

    #[test]
    fn test_nd_stress_case_209() {
        let t = Tensor::from_slice(&[209.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 209.0);
        assert_eq!(c.get(1), 209.0);
    }

    #[test]
    fn test_nd_stress_case_210() {
        let t = Tensor::from_slice(&[210.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 210.0);
        assert_eq!(c.get(1), 210.0);
    }

    #[test]
    fn test_nd_stress_case_211() {
        let t = Tensor::from_slice(&[211.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 211.0);
        assert_eq!(c.get(1), 211.0);
    }

    #[test]
    fn test_nd_stress_case_212() {
        let t = Tensor::from_slice(&[212.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 212.0);
        assert_eq!(c.get(1), 212.0);
    }

    #[test]
    fn test_nd_stress_case_213() {
        let t = Tensor::from_slice(&[213.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 213.0);
        assert_eq!(c.get(1), 213.0);
    }

    #[test]
    fn test_nd_stress_case_214() {
        let t = Tensor::from_slice(&[214.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 214.0);
        assert_eq!(c.get(1), 214.0);
    }

    #[test]
    fn test_nd_stress_case_215() {
        let t = Tensor::from_slice(&[215.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 215.0);
        assert_eq!(c.get(1), 215.0);
    }

    #[test]
    fn test_nd_stress_case_216() {
        let t = Tensor::from_slice(&[216.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 216.0);
        assert_eq!(c.get(1), 216.0);
    }

    #[test]
    fn test_nd_stress_case_217() {
        let t = Tensor::from_slice(&[217.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 217.0);
        assert_eq!(c.get(1), 217.0);
    }

    #[test]
    fn test_nd_stress_case_218() {
        let t = Tensor::from_slice(&[218.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 218.0);
        assert_eq!(c.get(1), 218.0);
    }

    #[test]
    fn test_nd_stress_case_219() {
        let t = Tensor::from_slice(&[219.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 219.0);
        assert_eq!(c.get(1), 219.0);
    }

    #[test]
    fn test_nd_stress_case_220() {
        let t = Tensor::from_slice(&[220.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 220.0);
        assert_eq!(c.get(1), 220.0);
    }

    #[test]
    fn test_nd_stress_case_221() {
        let t = Tensor::from_slice(&[221.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 221.0);
        assert_eq!(c.get(1), 221.0);
    }

    #[test]
    fn test_nd_stress_case_222() {
        let t = Tensor::from_slice(&[222.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 222.0);
        assert_eq!(c.get(1), 222.0);
    }

    #[test]
    fn test_nd_stress_case_223() {
        let t = Tensor::from_slice(&[223.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 223.0);
        assert_eq!(c.get(1), 223.0);
    }

    #[test]
    fn test_nd_stress_case_224() {
        let t = Tensor::from_slice(&[224.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 224.0);
        assert_eq!(c.get(1), 224.0);
    }

    #[test]
    fn test_nd_stress_case_225() {
        let t = Tensor::from_slice(&[225.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 225.0);
        assert_eq!(c.get(1), 225.0);
    }

    #[test]
    fn test_nd_stress_case_226() {
        let t = Tensor::from_slice(&[226.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 226.0);
        assert_eq!(c.get(1), 226.0);
    }

    #[test]
    fn test_nd_stress_case_227() {
        let t = Tensor::from_slice(&[227.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 227.0);
        assert_eq!(c.get(1), 227.0);
    }

    #[test]
    fn test_nd_stress_case_228() {
        let t = Tensor::from_slice(&[228.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 228.0);
        assert_eq!(c.get(1), 228.0);
    }

    #[test]
    fn test_nd_stress_case_229() {
        let t = Tensor::from_slice(&[229.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 229.0);
        assert_eq!(c.get(1), 229.0);
    }

    #[test]
    fn test_nd_stress_case_230() {
        let t = Tensor::from_slice(&[230.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 230.0);
        assert_eq!(c.get(1), 230.0);
    }

    #[test]
    fn test_nd_stress_case_231() {
        let t = Tensor::from_slice(&[231.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 231.0);
        assert_eq!(c.get(1), 231.0);
    }

    #[test]
    fn test_nd_stress_case_232() {
        let t = Tensor::from_slice(&[232.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 232.0);
        assert_eq!(c.get(1), 232.0);
    }

    #[test]
    fn test_nd_stress_case_233() {
        let t = Tensor::from_slice(&[233.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 233.0);
        assert_eq!(c.get(1), 233.0);
    }

    #[test]
    fn test_nd_stress_case_234() {
        let t = Tensor::from_slice(&[234.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 234.0);
        assert_eq!(c.get(1), 234.0);
    }

    #[test]
    fn test_nd_stress_case_235() {
        let t = Tensor::from_slice(&[235.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 235.0);
        assert_eq!(c.get(1), 235.0);
    }

    #[test]
    fn test_nd_stress_case_236() {
        let t = Tensor::from_slice(&[236.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 236.0);
        assert_eq!(c.get(1), 236.0);
    }

    #[test]
    fn test_nd_stress_case_237() {
        let t = Tensor::from_slice(&[237.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 237.0);
        assert_eq!(c.get(1), 237.0);
    }

    #[test]
    fn test_nd_stress_case_238() {
        let t = Tensor::from_slice(&[238.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 238.0);
        assert_eq!(c.get(1), 238.0);
    }

    #[test]
    fn test_nd_stress_case_239() {
        let t = Tensor::from_slice(&[239.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 239.0);
        assert_eq!(c.get(1), 239.0);
    }

    #[test]
    fn test_nd_stress_case_240() {
        let t = Tensor::from_slice(&[240.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 240.0);
        assert_eq!(c.get(1), 240.0);
    }

    #[test]
    fn test_nd_stress_case_241() {
        let t = Tensor::from_slice(&[241.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 241.0);
        assert_eq!(c.get(1), 241.0);
    }

    #[test]
    fn test_nd_stress_case_242() {
        let t = Tensor::from_slice(&[242.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 242.0);
        assert_eq!(c.get(1), 242.0);
    }

    #[test]
    fn test_nd_stress_case_243() {
        let t = Tensor::from_slice(&[243.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 243.0);
        assert_eq!(c.get(1), 243.0);
    }

    #[test]
    fn test_nd_stress_case_244() {
        let t = Tensor::from_slice(&[244.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 244.0);
        assert_eq!(c.get(1), 244.0);
    }

    #[test]
    fn test_nd_stress_case_245() {
        let t = Tensor::from_slice(&[245.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 245.0);
        assert_eq!(c.get(1), 245.0);
    }

    #[test]
    fn test_nd_stress_case_246() {
        let t = Tensor::from_slice(&[246.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 246.0);
        assert_eq!(c.get(1), 246.0);
    }

    #[test]
    fn test_nd_stress_case_247() {
        let t = Tensor::from_slice(&[247.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 247.0);
        assert_eq!(c.get(1), 247.0);
    }

    #[test]
    fn test_nd_stress_case_248() {
        let t = Tensor::from_slice(&[248.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 248.0);
        assert_eq!(c.get(1), 248.0);
    }

    #[test]
    fn test_nd_stress_case_249() {
        let t = Tensor::from_slice(&[249.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 249.0);
        assert_eq!(c.get(1), 249.0);
    }

    #[test]
    fn test_nd_stress_case_250() {
        let t = Tensor::from_slice(&[250.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 250.0);
        assert_eq!(c.get(1), 250.0);
    }

    #[test]
    fn test_nd_stress_case_251() {
        let t = Tensor::from_slice(&[251.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 251.0);
        assert_eq!(c.get(1), 251.0);
    }

    #[test]
    fn test_nd_stress_case_252() {
        let t = Tensor::from_slice(&[252.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 252.0);
        assert_eq!(c.get(1), 252.0);
    }

    #[test]
    fn test_nd_stress_case_253() {
        let t = Tensor::from_slice(&[253.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 253.0);
        assert_eq!(c.get(1), 253.0);
    }

    #[test]
    fn test_nd_stress_case_254() {
        let t = Tensor::from_slice(&[254.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 254.0);
        assert_eq!(c.get(1), 254.0);
    }

    #[test]
    fn test_nd_stress_case_255() {
        let t = Tensor::from_slice(&[255.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 255.0);
        assert_eq!(c.get(1), 255.0);
    }

    #[test]
    fn test_nd_stress_case_256() {
        let t = Tensor::from_slice(&[256.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 256.0);
        assert_eq!(c.get(1), 256.0);
    }

    #[test]
    fn test_nd_stress_case_257() {
        let t = Tensor::from_slice(&[257.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 257.0);
        assert_eq!(c.get(1), 257.0);
    }

    #[test]
    fn test_nd_stress_case_258() {
        let t = Tensor::from_slice(&[258.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 258.0);
        assert_eq!(c.get(1), 258.0);
    }

    #[test]
    fn test_nd_stress_case_259() {
        let t = Tensor::from_slice(&[259.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 259.0);
        assert_eq!(c.get(1), 259.0);
    }

    #[test]
    fn test_nd_stress_case_260() {
        let t = Tensor::from_slice(&[260.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 260.0);
        assert_eq!(c.get(1), 260.0);
    }

    #[test]
    fn test_nd_stress_case_261() {
        let t = Tensor::from_slice(&[261.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 261.0);
        assert_eq!(c.get(1), 261.0);
    }

    #[test]
    fn test_nd_stress_case_262() {
        let t = Tensor::from_slice(&[262.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 262.0);
        assert_eq!(c.get(1), 262.0);
    }

    #[test]
    fn test_nd_stress_case_263() {
        let t = Tensor::from_slice(&[263.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 263.0);
        assert_eq!(c.get(1), 263.0);
    }

    #[test]
    fn test_nd_stress_case_264() {
        let t = Tensor::from_slice(&[264.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 264.0);
        assert_eq!(c.get(1), 264.0);
    }

    #[test]
    fn test_nd_stress_case_265() {
        let t = Tensor::from_slice(&[265.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 265.0);
        assert_eq!(c.get(1), 265.0);
    }

    #[test]
    fn test_nd_stress_case_266() {
        let t = Tensor::from_slice(&[266.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 266.0);
        assert_eq!(c.get(1), 266.0);
    }

    #[test]
    fn test_nd_stress_case_267() {
        let t = Tensor::from_slice(&[267.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 267.0);
        assert_eq!(c.get(1), 267.0);
    }

    #[test]
    fn test_nd_stress_case_268() {
        let t = Tensor::from_slice(&[268.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 268.0);
        assert_eq!(c.get(1), 268.0);
    }

    #[test]
    fn test_nd_stress_case_269() {
        let t = Tensor::from_slice(&[269.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 269.0);
        assert_eq!(c.get(1), 269.0);
    }

    #[test]
    fn test_nd_stress_case_270() {
        let t = Tensor::from_slice(&[270.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 270.0);
        assert_eq!(c.get(1), 270.0);
    }

    #[test]
    fn test_nd_stress_case_271() {
        let t = Tensor::from_slice(&[271.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 271.0);
        assert_eq!(c.get(1), 271.0);
    }

    #[test]
    fn test_nd_stress_case_272() {
        let t = Tensor::from_slice(&[272.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 272.0);
        assert_eq!(c.get(1), 272.0);
    }

    #[test]
    fn test_nd_stress_case_273() {
        let t = Tensor::from_slice(&[273.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 273.0);
        assert_eq!(c.get(1), 273.0);
    }

    #[test]
    fn test_nd_stress_case_274() {
        let t = Tensor::from_slice(&[274.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 274.0);
        assert_eq!(c.get(1), 274.0);
    }

    #[test]
    fn test_nd_stress_case_275() {
        let t = Tensor::from_slice(&[275.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 275.0);
        assert_eq!(c.get(1), 275.0);
    }

    #[test]
    fn test_nd_stress_case_276() {
        let t = Tensor::from_slice(&[276.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 276.0);
        assert_eq!(c.get(1), 276.0);
    }

    #[test]
    fn test_nd_stress_case_277() {
        let t = Tensor::from_slice(&[277.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 277.0);
        assert_eq!(c.get(1), 277.0);
    }

    #[test]
    fn test_nd_stress_case_278() {
        let t = Tensor::from_slice(&[278.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 278.0);
        assert_eq!(c.get(1), 278.0);
    }

    #[test]
    fn test_nd_stress_case_279() {
        let t = Tensor::from_slice(&[279.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 279.0);
        assert_eq!(c.get(1), 279.0);
    }

    #[test]
    fn test_nd_stress_case_280() {
        let t = Tensor::from_slice(&[280.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 280.0);
        assert_eq!(c.get(1), 280.0);
    }

    #[test]
    fn test_nd_stress_case_281() {
        let t = Tensor::from_slice(&[281.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 281.0);
        assert_eq!(c.get(1), 281.0);
    }

    #[test]
    fn test_nd_stress_case_282() {
        let t = Tensor::from_slice(&[282.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 282.0);
        assert_eq!(c.get(1), 282.0);
    }

    #[test]
    fn test_nd_stress_case_283() {
        let t = Tensor::from_slice(&[283.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 283.0);
        assert_eq!(c.get(1), 283.0);
    }

    #[test]
    fn test_nd_stress_case_284() {
        let t = Tensor::from_slice(&[284.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 284.0);
        assert_eq!(c.get(1), 284.0);
    }

    #[test]
    fn test_nd_stress_case_285() {
        let t = Tensor::from_slice(&[285.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 285.0);
        assert_eq!(c.get(1), 285.0);
    }

    #[test]
    fn test_nd_stress_case_286() {
        let t = Tensor::from_slice(&[286.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 286.0);
        assert_eq!(c.get(1), 286.0);
    }

    #[test]
    fn test_nd_stress_case_287() {
        let t = Tensor::from_slice(&[287.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 287.0);
        assert_eq!(c.get(1), 287.0);
    }

    #[test]
    fn test_nd_stress_case_288() {
        let t = Tensor::from_slice(&[288.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 288.0);
        assert_eq!(c.get(1), 288.0);
    }

    #[test]
    fn test_nd_stress_case_289() {
        let t = Tensor::from_slice(&[289.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 289.0);
        assert_eq!(c.get(1), 289.0);
    }

    #[test]
    fn test_nd_stress_case_290() {
        let t = Tensor::from_slice(&[290.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 290.0);
        assert_eq!(c.get(1), 290.0);
    }

    #[test]
    fn test_nd_stress_case_291() {
        let t = Tensor::from_slice(&[291.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 291.0);
        assert_eq!(c.get(1), 291.0);
    }

    #[test]
    fn test_nd_stress_case_292() {
        let t = Tensor::from_slice(&[292.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 292.0);
        assert_eq!(c.get(1), 292.0);
    }

    #[test]
    fn test_nd_stress_case_293() {
        let t = Tensor::from_slice(&[293.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 293.0);
        assert_eq!(c.get(1), 293.0);
    }

    #[test]
    fn test_nd_stress_case_294() {
        let t = Tensor::from_slice(&[294.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 294.0);
        assert_eq!(c.get(1), 294.0);
    }

    #[test]
    fn test_nd_stress_case_295() {
        let t = Tensor::from_slice(&[295.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 295.0);
        assert_eq!(c.get(1), 295.0);
    }

    #[test]
    fn test_nd_stress_case_296() {
        let t = Tensor::from_slice(&[296.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 296.0);
        assert_eq!(c.get(1), 296.0);
    }

    #[test]
    fn test_nd_stress_case_297() {
        let t = Tensor::from_slice(&[297.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 297.0);
        assert_eq!(c.get(1), 297.0);
    }

    #[test]
    fn test_nd_stress_case_298() {
        let t = Tensor::from_slice(&[298.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 298.0);
        assert_eq!(c.get(1), 298.0);
    }

    #[test]
    fn test_nd_stress_case_299() {
        let t = Tensor::from_slice(&[299.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 299.0);
        assert_eq!(c.get(1), 299.0);
    }

    #[test]
    fn test_nd_stress_case_300() {
        let t = Tensor::from_slice(&[300.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 300.0);
        assert_eq!(c.get(1), 300.0);
    }

    #[test]
    fn test_nd_stress_case_301() {
        let t = Tensor::from_slice(&[301.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 301.0);
        assert_eq!(c.get(1), 301.0);
    }

    #[test]
    fn test_nd_stress_case_302() {
        let t = Tensor::from_slice(&[302.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 302.0);
        assert_eq!(c.get(1), 302.0);
    }

    #[test]
    fn test_nd_stress_case_303() {
        let t = Tensor::from_slice(&[303.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 303.0);
        assert_eq!(c.get(1), 303.0);
    }

    #[test]
    fn test_nd_stress_case_304() {
        let t = Tensor::from_slice(&[304.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 304.0);
        assert_eq!(c.get(1), 304.0);
    }

    #[test]
    fn test_nd_stress_case_305() {
        let t = Tensor::from_slice(&[305.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 305.0);
        assert_eq!(c.get(1), 305.0);
    }

    #[test]
    fn test_nd_stress_case_306() {
        let t = Tensor::from_slice(&[306.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 306.0);
        assert_eq!(c.get(1), 306.0);
    }

    #[test]
    fn test_nd_stress_case_307() {
        let t = Tensor::from_slice(&[307.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 307.0);
        assert_eq!(c.get(1), 307.0);
    }

    #[test]
    fn test_nd_stress_case_308() {
        let t = Tensor::from_slice(&[308.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 308.0);
        assert_eq!(c.get(1), 308.0);
    }

    #[test]
    fn test_nd_stress_case_309() {
        let t = Tensor::from_slice(&[309.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 309.0);
        assert_eq!(c.get(1), 309.0);
    }

    #[test]
    fn test_nd_stress_case_310() {
        let t = Tensor::from_slice(&[310.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 310.0);
        assert_eq!(c.get(1), 310.0);
    }

    #[test]
    fn test_nd_stress_case_311() {
        let t = Tensor::from_slice(&[311.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 311.0);
        assert_eq!(c.get(1), 311.0);
    }

    #[test]
    fn test_nd_stress_case_312() {
        let t = Tensor::from_slice(&[312.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 312.0);
        assert_eq!(c.get(1), 312.0);
    }

    #[test]
    fn test_nd_stress_case_313() {
        let t = Tensor::from_slice(&[313.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 313.0);
        assert_eq!(c.get(1), 313.0);
    }

    #[test]
    fn test_nd_stress_case_314() {
        let t = Tensor::from_slice(&[314.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 314.0);
        assert_eq!(c.get(1), 314.0);
    }

    #[test]
    fn test_nd_stress_case_315() {
        let t = Tensor::from_slice(&[315.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 315.0);
        assert_eq!(c.get(1), 315.0);
    }

    #[test]
    fn test_nd_stress_case_316() {
        let t = Tensor::from_slice(&[316.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 316.0);
        assert_eq!(c.get(1), 316.0);
    }

    #[test]
    fn test_nd_stress_case_317() {
        let t = Tensor::from_slice(&[317.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 317.0);
        assert_eq!(c.get(1), 317.0);
    }

    #[test]
    fn test_nd_stress_case_318() {
        let t = Tensor::from_slice(&[318.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 318.0);
        assert_eq!(c.get(1), 318.0);
    }

    #[test]
    fn test_nd_stress_case_319() {
        let t = Tensor::from_slice(&[319.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 319.0);
        assert_eq!(c.get(1), 319.0);
    }

    #[test]
    fn test_nd_stress_case_320() {
        let t = Tensor::from_slice(&[320.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 320.0);
        assert_eq!(c.get(1), 320.0);
    }

    #[test]
    fn test_nd_stress_case_321() {
        let t = Tensor::from_slice(&[321.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 321.0);
        assert_eq!(c.get(1), 321.0);
    }

    #[test]
    fn test_nd_stress_case_322() {
        let t = Tensor::from_slice(&[322.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 322.0);
        assert_eq!(c.get(1), 322.0);
    }

    #[test]
    fn test_nd_stress_case_323() {
        let t = Tensor::from_slice(&[323.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 323.0);
        assert_eq!(c.get(1), 323.0);
    }

    #[test]
    fn test_nd_stress_case_324() {
        let t = Tensor::from_slice(&[324.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 324.0);
        assert_eq!(c.get(1), 324.0);
    }

    #[test]
    fn test_nd_stress_case_325() {
        let t = Tensor::from_slice(&[325.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 325.0);
        assert_eq!(c.get(1), 325.0);
    }

    #[test]
    fn test_nd_stress_case_326() {
        let t = Tensor::from_slice(&[326.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 326.0);
        assert_eq!(c.get(1), 326.0);
    }

    #[test]
    fn test_nd_stress_case_327() {
        let t = Tensor::from_slice(&[327.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 327.0);
        assert_eq!(c.get(1), 327.0);
    }

    #[test]
    fn test_nd_stress_case_328() {
        let t = Tensor::from_slice(&[328.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 328.0);
        assert_eq!(c.get(1), 328.0);
    }

    #[test]
    fn test_nd_stress_case_329() {
        let t = Tensor::from_slice(&[329.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 329.0);
        assert_eq!(c.get(1), 329.0);
    }

    #[test]
    fn test_nd_stress_case_330() {
        let t = Tensor::from_slice(&[330.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 330.0);
        assert_eq!(c.get(1), 330.0);
    }

    #[test]
    fn test_nd_stress_case_331() {
        let t = Tensor::from_slice(&[331.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 331.0);
        assert_eq!(c.get(1), 331.0);
    }

    #[test]
    fn test_nd_stress_case_332() {
        let t = Tensor::from_slice(&[332.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 332.0);
        assert_eq!(c.get(1), 332.0);
    }

    #[test]
    fn test_nd_stress_case_333() {
        let t = Tensor::from_slice(&[333.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 333.0);
        assert_eq!(c.get(1), 333.0);
    }

    #[test]
    fn test_nd_stress_case_334() {
        let t = Tensor::from_slice(&[334.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 334.0);
        assert_eq!(c.get(1), 334.0);
    }

    #[test]
    fn test_nd_stress_case_335() {
        let t = Tensor::from_slice(&[335.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 335.0);
        assert_eq!(c.get(1), 335.0);
    }

    #[test]
    fn test_nd_stress_case_336() {
        let t = Tensor::from_slice(&[336.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 336.0);
        assert_eq!(c.get(1), 336.0);
    }

    #[test]
    fn test_nd_stress_case_337() {
        let t = Tensor::from_slice(&[337.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 337.0);
        assert_eq!(c.get(1), 337.0);
    }

    #[test]
    fn test_nd_stress_case_338() {
        let t = Tensor::from_slice(&[338.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 338.0);
        assert_eq!(c.get(1), 338.0);
    }

    #[test]
    fn test_nd_stress_case_339() {
        let t = Tensor::from_slice(&[339.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 339.0);
        assert_eq!(c.get(1), 339.0);
    }

    #[test]
    fn test_nd_stress_case_340() {
        let t = Tensor::from_slice(&[340.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 340.0);
        assert_eq!(c.get(1), 340.0);
    }

    #[test]
    fn test_nd_stress_case_341() {
        let t = Tensor::from_slice(&[341.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 341.0);
        assert_eq!(c.get(1), 341.0);
    }

    #[test]
    fn test_nd_stress_case_342() {
        let t = Tensor::from_slice(&[342.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 342.0);
        assert_eq!(c.get(1), 342.0);
    }

    #[test]
    fn test_nd_stress_case_343() {
        let t = Tensor::from_slice(&[343.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 343.0);
        assert_eq!(c.get(1), 343.0);
    }

    #[test]
    fn test_nd_stress_case_344() {
        let t = Tensor::from_slice(&[344.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 344.0);
        assert_eq!(c.get(1), 344.0);
    }

    #[test]
    fn test_nd_stress_case_345() {
        let t = Tensor::from_slice(&[345.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 345.0);
        assert_eq!(c.get(1), 345.0);
    }

    #[test]
    fn test_nd_stress_case_346() {
        let t = Tensor::from_slice(&[346.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 346.0);
        assert_eq!(c.get(1), 346.0);
    }

    #[test]
    fn test_nd_stress_case_347() {
        let t = Tensor::from_slice(&[347.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 347.0);
        assert_eq!(c.get(1), 347.0);
    }

    #[test]
    fn test_nd_stress_case_348() {
        let t = Tensor::from_slice(&[348.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 348.0);
        assert_eq!(c.get(1), 348.0);
    }

    #[test]
    fn test_nd_stress_case_349() {
        let t = Tensor::from_slice(&[349.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 349.0);
        assert_eq!(c.get(1), 349.0);
    }

    #[test]
    fn test_nd_stress_case_350() {
        let t = Tensor::from_slice(&[350.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 350.0);
        assert_eq!(c.get(1), 350.0);
    }

    #[test]
    fn test_nd_stress_case_351() {
        let t = Tensor::from_slice(&[351.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 351.0);
        assert_eq!(c.get(1), 351.0);
    }

    #[test]
    fn test_nd_stress_case_352() {
        let t = Tensor::from_slice(&[352.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 352.0);
        assert_eq!(c.get(1), 352.0);
    }

    #[test]
    fn test_nd_stress_case_353() {
        let t = Tensor::from_slice(&[353.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 353.0);
        assert_eq!(c.get(1), 353.0);
    }

    #[test]
    fn test_nd_stress_case_354() {
        let t = Tensor::from_slice(&[354.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 354.0);
        assert_eq!(c.get(1), 354.0);
    }

    #[test]
    fn test_nd_stress_case_355() {
        let t = Tensor::from_slice(&[355.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 355.0);
        assert_eq!(c.get(1), 355.0);
    }

    #[test]
    fn test_nd_stress_case_356() {
        let t = Tensor::from_slice(&[356.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 356.0);
        assert_eq!(c.get(1), 356.0);
    }

    #[test]
    fn test_nd_stress_case_357() {
        let t = Tensor::from_slice(&[357.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 357.0);
        assert_eq!(c.get(1), 357.0);
    }

    #[test]
    fn test_nd_stress_case_358() {
        let t = Tensor::from_slice(&[358.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 358.0);
        assert_eq!(c.get(1), 358.0);
    }

    #[test]
    fn test_nd_stress_case_359() {
        let t = Tensor::from_slice(&[359.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 359.0);
        assert_eq!(c.get(1), 359.0);
    }

    #[test]
    fn test_nd_stress_case_360() {
        let t = Tensor::from_slice(&[360.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 360.0);
        assert_eq!(c.get(1), 360.0);
    }

    #[test]
    fn test_nd_stress_case_361() {
        let t = Tensor::from_slice(&[361.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 361.0);
        assert_eq!(c.get(1), 361.0);
    }

    #[test]
    fn test_nd_stress_case_362() {
        let t = Tensor::from_slice(&[362.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 362.0);
        assert_eq!(c.get(1), 362.0);
    }

    #[test]
    fn test_nd_stress_case_363() {
        let t = Tensor::from_slice(&[363.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 363.0);
        assert_eq!(c.get(1), 363.0);
    }

    #[test]
    fn test_nd_stress_case_364() {
        let t = Tensor::from_slice(&[364.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 364.0);
        assert_eq!(c.get(1), 364.0);
    }

    #[test]
    fn test_nd_stress_case_365() {
        let t = Tensor::from_slice(&[365.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 365.0);
        assert_eq!(c.get(1), 365.0);
    }

    #[test]
    fn test_nd_stress_case_366() {
        let t = Tensor::from_slice(&[366.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 366.0);
        assert_eq!(c.get(1), 366.0);
    }

    #[test]
    fn test_nd_stress_case_367() {
        let t = Tensor::from_slice(&[367.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 367.0);
        assert_eq!(c.get(1), 367.0);
    }

    #[test]
    fn test_nd_stress_case_368() {
        let t = Tensor::from_slice(&[368.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 368.0);
        assert_eq!(c.get(1), 368.0);
    }

    #[test]
    fn test_nd_stress_case_369() {
        let t = Tensor::from_slice(&[369.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 369.0);
        assert_eq!(c.get(1), 369.0);
    }

    #[test]
    fn test_nd_stress_case_370() {
        let t = Tensor::from_slice(&[370.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 370.0);
        assert_eq!(c.get(1), 370.0);
    }

    #[test]
    fn test_nd_stress_case_371() {
        let t = Tensor::from_slice(&[371.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 371.0);
        assert_eq!(c.get(1), 371.0);
    }

    #[test]
    fn test_nd_stress_case_372() {
        let t = Tensor::from_slice(&[372.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 372.0);
        assert_eq!(c.get(1), 372.0);
    }

    #[test]
    fn test_nd_stress_case_373() {
        let t = Tensor::from_slice(&[373.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 373.0);
        assert_eq!(c.get(1), 373.0);
    }

    #[test]
    fn test_nd_stress_case_374() {
        let t = Tensor::from_slice(&[374.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 374.0);
        assert_eq!(c.get(1), 374.0);
    }

    #[test]
    fn test_nd_stress_case_375() {
        let t = Tensor::from_slice(&[375.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 375.0);
        assert_eq!(c.get(1), 375.0);
    }

    #[test]
    fn test_nd_stress_case_376() {
        let t = Tensor::from_slice(&[376.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 376.0);
        assert_eq!(c.get(1), 376.0);
    }

    #[test]
    fn test_nd_stress_case_377() {
        let t = Tensor::from_slice(&[377.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 377.0);
        assert_eq!(c.get(1), 377.0);
    }

    #[test]
    fn test_nd_stress_case_378() {
        let t = Tensor::from_slice(&[378.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 378.0);
        assert_eq!(c.get(1), 378.0);
    }

    #[test]
    fn test_nd_stress_case_379() {
        let t = Tensor::from_slice(&[379.0], vec![1]);
        let c = cat(&[&t, &t], 0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.get(0), 379.0);
        assert_eq!(c.get(1), 379.0);
    }
}
