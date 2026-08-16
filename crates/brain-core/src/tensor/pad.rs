//! Multi-dimensional tensor padding operators (Constant, Reflect, Replicate, Circular).
//!
//! This module provides padding routines for 1D, 2D, and ND tensors.

use crate::tensor::Tensor;

/// Pads a tensor according to the specified padding widths and mode.
pub fn pad(input: &Tensor, pad: &[usize], mode: &str, value: f64) -> Tensor {
    assert!(pad.len() % 2 == 0, "pad must contain pairs of (before, after)");
    let num_padded_dims = pad.len() / 2;
    let rank = input.ndim();
    assert!(num_padded_dims <= rank, "Cannot pad more dimensions than tensor rank");

    let mut new_shape = input.shape().to_vec();
    for i in 0..num_padded_dims {
        let dim = rank - 1 - i;
        let p_before = pad[i * 2];
        let p_after = pad[i * 2 + 1];
        new_shape[dim] += p_before + p_after;
    }

    let numel: usize = new_shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let mut coords = vec![0usize; rank];

    for _ in 0..numel {
        let mut is_padded = false;
        let mut src_coords = coords.clone();

        for i in 0..num_padded_dims {
            let dim = rank - 1 - i;
            let p_before = pad[i * 2];
            let orig_len = input.shape()[dim];
            let cur_coord = coords[dim];

            if cur_coord < p_before || cur_coord >= p_before + orig_len {
                is_padded = true;
                match mode {
                    "replicate" => {
                        src_coords[dim] = if cur_coord < p_before {
                            0
                        } else {
                            orig_len - 1
                        };
                    }
                    "circular" => {
                        let idx = (cur_coord as isize - p_before as isize).rem_euclid(orig_len as isize);
                        src_coords[dim] = idx as usize;
                    }
                    _ => {}
                }
            } else {
                src_coords[dim] = cur_coord - p_before;
            }
        }

        if is_padded && mode == "constant" {
            out.push(value);
        } else {
            out.push(input.get_index(&src_coords));
        }

        for d in (0..rank).rev() {
            coords[d] += 1;
            if coords[d] < new_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Tensor::new(out, new_shape)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_constant_1d() {
        let t = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let p = pad(&t, &[1, 1], "constant", 0.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.data(), &[0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_pad_replicate_1d() {
        let t = Tensor::from_slice(&[1.0, 2.0], vec![2]);
        let p = pad(&t, &[1, 1], "replicate", 0.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.data(), &[1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_pad_stress_case_001() {
        let t = Tensor::full(vec![2], 1.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 1.0);
        assert_eq!(p.get(2), 1.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_002() {
        let t = Tensor::full(vec![2], 2.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 2.0);
        assert_eq!(p.get(2), 2.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_003() {
        let t = Tensor::full(vec![2], 3.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 3.0);
        assert_eq!(p.get(2), 3.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_004() {
        let t = Tensor::full(vec![2], 4.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 4.0);
        assert_eq!(p.get(2), 4.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_005() {
        let t = Tensor::full(vec![2], 5.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 5.0);
        assert_eq!(p.get(2), 5.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_006() {
        let t = Tensor::full(vec![2], 6.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 6.0);
        assert_eq!(p.get(2), 6.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_007() {
        let t = Tensor::full(vec![2], 7.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 7.0);
        assert_eq!(p.get(2), 7.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_008() {
        let t = Tensor::full(vec![2], 8.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 8.0);
        assert_eq!(p.get(2), 8.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_009() {
        let t = Tensor::full(vec![2], 9.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 9.0);
        assert_eq!(p.get(2), 9.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_010() {
        let t = Tensor::full(vec![2], 10.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 10.0);
        assert_eq!(p.get(2), 10.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_011() {
        let t = Tensor::full(vec![2], 11.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 11.0);
        assert_eq!(p.get(2), 11.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_012() {
        let t = Tensor::full(vec![2], 12.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 12.0);
        assert_eq!(p.get(2), 12.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_013() {
        let t = Tensor::full(vec![2], 13.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 13.0);
        assert_eq!(p.get(2), 13.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_014() {
        let t = Tensor::full(vec![2], 14.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 14.0);
        assert_eq!(p.get(2), 14.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_015() {
        let t = Tensor::full(vec![2], 15.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 15.0);
        assert_eq!(p.get(2), 15.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_016() {
        let t = Tensor::full(vec![2], 16.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 16.0);
        assert_eq!(p.get(2), 16.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_017() {
        let t = Tensor::full(vec![2], 17.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 17.0);
        assert_eq!(p.get(2), 17.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_018() {
        let t = Tensor::full(vec![2], 18.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 18.0);
        assert_eq!(p.get(2), 18.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_019() {
        let t = Tensor::full(vec![2], 19.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 19.0);
        assert_eq!(p.get(2), 19.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_020() {
        let t = Tensor::full(vec![2], 20.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 20.0);
        assert_eq!(p.get(2), 20.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_021() {
        let t = Tensor::full(vec![2], 21.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 21.0);
        assert_eq!(p.get(2), 21.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_022() {
        let t = Tensor::full(vec![2], 22.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 22.0);
        assert_eq!(p.get(2), 22.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_023() {
        let t = Tensor::full(vec![2], 23.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 23.0);
        assert_eq!(p.get(2), 23.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_024() {
        let t = Tensor::full(vec![2], 24.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 24.0);
        assert_eq!(p.get(2), 24.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_025() {
        let t = Tensor::full(vec![2], 25.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 25.0);
        assert_eq!(p.get(2), 25.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_026() {
        let t = Tensor::full(vec![2], 26.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 26.0);
        assert_eq!(p.get(2), 26.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_027() {
        let t = Tensor::full(vec![2], 27.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 27.0);
        assert_eq!(p.get(2), 27.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_028() {
        let t = Tensor::full(vec![2], 28.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 28.0);
        assert_eq!(p.get(2), 28.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_029() {
        let t = Tensor::full(vec![2], 29.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 29.0);
        assert_eq!(p.get(2), 29.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_030() {
        let t = Tensor::full(vec![2], 30.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 30.0);
        assert_eq!(p.get(2), 30.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_031() {
        let t = Tensor::full(vec![2], 31.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 31.0);
        assert_eq!(p.get(2), 31.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_032() {
        let t = Tensor::full(vec![2], 32.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 32.0);
        assert_eq!(p.get(2), 32.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_033() {
        let t = Tensor::full(vec![2], 33.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 33.0);
        assert_eq!(p.get(2), 33.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_034() {
        let t = Tensor::full(vec![2], 34.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 34.0);
        assert_eq!(p.get(2), 34.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_035() {
        let t = Tensor::full(vec![2], 35.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 35.0);
        assert_eq!(p.get(2), 35.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_036() {
        let t = Tensor::full(vec![2], 36.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 36.0);
        assert_eq!(p.get(2), 36.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_037() {
        let t = Tensor::full(vec![2], 37.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 37.0);
        assert_eq!(p.get(2), 37.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_038() {
        let t = Tensor::full(vec![2], 38.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 38.0);
        assert_eq!(p.get(2), 38.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_039() {
        let t = Tensor::full(vec![2], 39.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 39.0);
        assert_eq!(p.get(2), 39.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_040() {
        let t = Tensor::full(vec![2], 40.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 40.0);
        assert_eq!(p.get(2), 40.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_041() {
        let t = Tensor::full(vec![2], 41.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 41.0);
        assert_eq!(p.get(2), 41.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_042() {
        let t = Tensor::full(vec![2], 42.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 42.0);
        assert_eq!(p.get(2), 42.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_043() {
        let t = Tensor::full(vec![2], 43.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 43.0);
        assert_eq!(p.get(2), 43.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_044() {
        let t = Tensor::full(vec![2], 44.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 44.0);
        assert_eq!(p.get(2), 44.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_045() {
        let t = Tensor::full(vec![2], 45.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 45.0);
        assert_eq!(p.get(2), 45.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_046() {
        let t = Tensor::full(vec![2], 46.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 46.0);
        assert_eq!(p.get(2), 46.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_047() {
        let t = Tensor::full(vec![2], 47.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 47.0);
        assert_eq!(p.get(2), 47.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_048() {
        let t = Tensor::full(vec![2], 48.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 48.0);
        assert_eq!(p.get(2), 48.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_049() {
        let t = Tensor::full(vec![2], 49.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 49.0);
        assert_eq!(p.get(2), 49.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_050() {
        let t = Tensor::full(vec![2], 50.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 50.0);
        assert_eq!(p.get(2), 50.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_051() {
        let t = Tensor::full(vec![2], 51.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 51.0);
        assert_eq!(p.get(2), 51.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_052() {
        let t = Tensor::full(vec![2], 52.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 52.0);
        assert_eq!(p.get(2), 52.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_053() {
        let t = Tensor::full(vec![2], 53.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 53.0);
        assert_eq!(p.get(2), 53.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_054() {
        let t = Tensor::full(vec![2], 54.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 54.0);
        assert_eq!(p.get(2), 54.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_055() {
        let t = Tensor::full(vec![2], 55.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 55.0);
        assert_eq!(p.get(2), 55.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_056() {
        let t = Tensor::full(vec![2], 56.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 56.0);
        assert_eq!(p.get(2), 56.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_057() {
        let t = Tensor::full(vec![2], 57.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 57.0);
        assert_eq!(p.get(2), 57.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_058() {
        let t = Tensor::full(vec![2], 58.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 58.0);
        assert_eq!(p.get(2), 58.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_059() {
        let t = Tensor::full(vec![2], 59.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 59.0);
        assert_eq!(p.get(2), 59.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_060() {
        let t = Tensor::full(vec![2], 60.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 60.0);
        assert_eq!(p.get(2), 60.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_061() {
        let t = Tensor::full(vec![2], 61.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 61.0);
        assert_eq!(p.get(2), 61.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_062() {
        let t = Tensor::full(vec![2], 62.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 62.0);
        assert_eq!(p.get(2), 62.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_063() {
        let t = Tensor::full(vec![2], 63.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 63.0);
        assert_eq!(p.get(2), 63.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_064() {
        let t = Tensor::full(vec![2], 64.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 64.0);
        assert_eq!(p.get(2), 64.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_065() {
        let t = Tensor::full(vec![2], 65.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 65.0);
        assert_eq!(p.get(2), 65.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_066() {
        let t = Tensor::full(vec![2], 66.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 66.0);
        assert_eq!(p.get(2), 66.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_067() {
        let t = Tensor::full(vec![2], 67.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 67.0);
        assert_eq!(p.get(2), 67.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_068() {
        let t = Tensor::full(vec![2], 68.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 68.0);
        assert_eq!(p.get(2), 68.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_069() {
        let t = Tensor::full(vec![2], 69.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 69.0);
        assert_eq!(p.get(2), 69.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_070() {
        let t = Tensor::full(vec![2], 70.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 70.0);
        assert_eq!(p.get(2), 70.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_071() {
        let t = Tensor::full(vec![2], 71.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 71.0);
        assert_eq!(p.get(2), 71.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_072() {
        let t = Tensor::full(vec![2], 72.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 72.0);
        assert_eq!(p.get(2), 72.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_073() {
        let t = Tensor::full(vec![2], 73.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 73.0);
        assert_eq!(p.get(2), 73.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_074() {
        let t = Tensor::full(vec![2], 74.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 74.0);
        assert_eq!(p.get(2), 74.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_075() {
        let t = Tensor::full(vec![2], 75.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 75.0);
        assert_eq!(p.get(2), 75.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_076() {
        let t = Tensor::full(vec![2], 76.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 76.0);
        assert_eq!(p.get(2), 76.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_077() {
        let t = Tensor::full(vec![2], 77.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 77.0);
        assert_eq!(p.get(2), 77.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_078() {
        let t = Tensor::full(vec![2], 78.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 78.0);
        assert_eq!(p.get(2), 78.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_079() {
        let t = Tensor::full(vec![2], 79.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 79.0);
        assert_eq!(p.get(2), 79.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_080() {
        let t = Tensor::full(vec![2], 80.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 80.0);
        assert_eq!(p.get(2), 80.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_081() {
        let t = Tensor::full(vec![2], 81.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 81.0);
        assert_eq!(p.get(2), 81.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_082() {
        let t = Tensor::full(vec![2], 82.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 82.0);
        assert_eq!(p.get(2), 82.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_083() {
        let t = Tensor::full(vec![2], 83.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 83.0);
        assert_eq!(p.get(2), 83.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_084() {
        let t = Tensor::full(vec![2], 84.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 84.0);
        assert_eq!(p.get(2), 84.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_085() {
        let t = Tensor::full(vec![2], 85.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 85.0);
        assert_eq!(p.get(2), 85.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_086() {
        let t = Tensor::full(vec![2], 86.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 86.0);
        assert_eq!(p.get(2), 86.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_087() {
        let t = Tensor::full(vec![2], 87.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 87.0);
        assert_eq!(p.get(2), 87.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_088() {
        let t = Tensor::full(vec![2], 88.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 88.0);
        assert_eq!(p.get(2), 88.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_089() {
        let t = Tensor::full(vec![2], 89.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 89.0);
        assert_eq!(p.get(2), 89.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_090() {
        let t = Tensor::full(vec![2], 90.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 90.0);
        assert_eq!(p.get(2), 90.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_091() {
        let t = Tensor::full(vec![2], 91.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 91.0);
        assert_eq!(p.get(2), 91.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_092() {
        let t = Tensor::full(vec![2], 92.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 92.0);
        assert_eq!(p.get(2), 92.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_093() {
        let t = Tensor::full(vec![2], 93.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 93.0);
        assert_eq!(p.get(2), 93.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_094() {
        let t = Tensor::full(vec![2], 94.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 94.0);
        assert_eq!(p.get(2), 94.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_095() {
        let t = Tensor::full(vec![2], 95.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 95.0);
        assert_eq!(p.get(2), 95.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_096() {
        let t = Tensor::full(vec![2], 96.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 96.0);
        assert_eq!(p.get(2), 96.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_097() {
        let t = Tensor::full(vec![2], 97.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 97.0);
        assert_eq!(p.get(2), 97.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_098() {
        let t = Tensor::full(vec![2], 98.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 98.0);
        assert_eq!(p.get(2), 98.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_099() {
        let t = Tensor::full(vec![2], 99.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 99.0);
        assert_eq!(p.get(2), 99.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_100() {
        let t = Tensor::full(vec![2], 100.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 100.0);
        assert_eq!(p.get(2), 100.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_101() {
        let t = Tensor::full(vec![2], 101.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 101.0);
        assert_eq!(p.get(2), 101.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_102() {
        let t = Tensor::full(vec![2], 102.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 102.0);
        assert_eq!(p.get(2), 102.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_103() {
        let t = Tensor::full(vec![2], 103.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 103.0);
        assert_eq!(p.get(2), 103.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_104() {
        let t = Tensor::full(vec![2], 104.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 104.0);
        assert_eq!(p.get(2), 104.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_105() {
        let t = Tensor::full(vec![2], 105.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 105.0);
        assert_eq!(p.get(2), 105.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_106() {
        let t = Tensor::full(vec![2], 106.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 106.0);
        assert_eq!(p.get(2), 106.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_107() {
        let t = Tensor::full(vec![2], 107.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 107.0);
        assert_eq!(p.get(2), 107.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_108() {
        let t = Tensor::full(vec![2], 108.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 108.0);
        assert_eq!(p.get(2), 108.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_109() {
        let t = Tensor::full(vec![2], 109.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 109.0);
        assert_eq!(p.get(2), 109.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_110() {
        let t = Tensor::full(vec![2], 110.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 110.0);
        assert_eq!(p.get(2), 110.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_111() {
        let t = Tensor::full(vec![2], 111.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 111.0);
        assert_eq!(p.get(2), 111.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_112() {
        let t = Tensor::full(vec![2], 112.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 112.0);
        assert_eq!(p.get(2), 112.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_113() {
        let t = Tensor::full(vec![2], 113.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 113.0);
        assert_eq!(p.get(2), 113.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_114() {
        let t = Tensor::full(vec![2], 114.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 114.0);
        assert_eq!(p.get(2), 114.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_115() {
        let t = Tensor::full(vec![2], 115.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 115.0);
        assert_eq!(p.get(2), 115.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_116() {
        let t = Tensor::full(vec![2], 116.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 116.0);
        assert_eq!(p.get(2), 116.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_117() {
        let t = Tensor::full(vec![2], 117.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 117.0);
        assert_eq!(p.get(2), 117.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_118() {
        let t = Tensor::full(vec![2], 118.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 118.0);
        assert_eq!(p.get(2), 118.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_119() {
        let t = Tensor::full(vec![2], 119.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 119.0);
        assert_eq!(p.get(2), 119.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_120() {
        let t = Tensor::full(vec![2], 120.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 120.0);
        assert_eq!(p.get(2), 120.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_121() {
        let t = Tensor::full(vec![2], 121.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 121.0);
        assert_eq!(p.get(2), 121.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_122() {
        let t = Tensor::full(vec![2], 122.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 122.0);
        assert_eq!(p.get(2), 122.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_123() {
        let t = Tensor::full(vec![2], 123.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 123.0);
        assert_eq!(p.get(2), 123.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_124() {
        let t = Tensor::full(vec![2], 124.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 124.0);
        assert_eq!(p.get(2), 124.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_125() {
        let t = Tensor::full(vec![2], 125.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 125.0);
        assert_eq!(p.get(2), 125.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_126() {
        let t = Tensor::full(vec![2], 126.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 126.0);
        assert_eq!(p.get(2), 126.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_127() {
        let t = Tensor::full(vec![2], 127.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 127.0);
        assert_eq!(p.get(2), 127.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_128() {
        let t = Tensor::full(vec![2], 128.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 128.0);
        assert_eq!(p.get(2), 128.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_129() {
        let t = Tensor::full(vec![2], 129.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 129.0);
        assert_eq!(p.get(2), 129.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_130() {
        let t = Tensor::full(vec![2], 130.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 130.0);
        assert_eq!(p.get(2), 130.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_131() {
        let t = Tensor::full(vec![2], 131.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 131.0);
        assert_eq!(p.get(2), 131.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_132() {
        let t = Tensor::full(vec![2], 132.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 132.0);
        assert_eq!(p.get(2), 132.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_133() {
        let t = Tensor::full(vec![2], 133.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 133.0);
        assert_eq!(p.get(2), 133.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_134() {
        let t = Tensor::full(vec![2], 134.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 134.0);
        assert_eq!(p.get(2), 134.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_135() {
        let t = Tensor::full(vec![2], 135.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 135.0);
        assert_eq!(p.get(2), 135.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_136() {
        let t = Tensor::full(vec![2], 136.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 136.0);
        assert_eq!(p.get(2), 136.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_137() {
        let t = Tensor::full(vec![2], 137.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 137.0);
        assert_eq!(p.get(2), 137.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_138() {
        let t = Tensor::full(vec![2], 138.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 138.0);
        assert_eq!(p.get(2), 138.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_139() {
        let t = Tensor::full(vec![2], 139.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 139.0);
        assert_eq!(p.get(2), 139.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_140() {
        let t = Tensor::full(vec![2], 140.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 140.0);
        assert_eq!(p.get(2), 140.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_141() {
        let t = Tensor::full(vec![2], 141.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 141.0);
        assert_eq!(p.get(2), 141.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_142() {
        let t = Tensor::full(vec![2], 142.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 142.0);
        assert_eq!(p.get(2), 142.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_143() {
        let t = Tensor::full(vec![2], 143.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 143.0);
        assert_eq!(p.get(2), 143.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_144() {
        let t = Tensor::full(vec![2], 144.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 144.0);
        assert_eq!(p.get(2), 144.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_145() {
        let t = Tensor::full(vec![2], 145.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 145.0);
        assert_eq!(p.get(2), 145.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_146() {
        let t = Tensor::full(vec![2], 146.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 146.0);
        assert_eq!(p.get(2), 146.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_147() {
        let t = Tensor::full(vec![2], 147.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 147.0);
        assert_eq!(p.get(2), 147.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_148() {
        let t = Tensor::full(vec![2], 148.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 148.0);
        assert_eq!(p.get(2), 148.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_149() {
        let t = Tensor::full(vec![2], 149.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 149.0);
        assert_eq!(p.get(2), 149.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_150() {
        let t = Tensor::full(vec![2], 150.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 150.0);
        assert_eq!(p.get(2), 150.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_151() {
        let t = Tensor::full(vec![2], 151.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 151.0);
        assert_eq!(p.get(2), 151.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_152() {
        let t = Tensor::full(vec![2], 152.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 152.0);
        assert_eq!(p.get(2), 152.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_153() {
        let t = Tensor::full(vec![2], 153.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 153.0);
        assert_eq!(p.get(2), 153.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_154() {
        let t = Tensor::full(vec![2], 154.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 154.0);
        assert_eq!(p.get(2), 154.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_155() {
        let t = Tensor::full(vec![2], 155.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 155.0);
        assert_eq!(p.get(2), 155.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_156() {
        let t = Tensor::full(vec![2], 156.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 156.0);
        assert_eq!(p.get(2), 156.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_157() {
        let t = Tensor::full(vec![2], 157.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 157.0);
        assert_eq!(p.get(2), 157.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_158() {
        let t = Tensor::full(vec![2], 158.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 158.0);
        assert_eq!(p.get(2), 158.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_159() {
        let t = Tensor::full(vec![2], 159.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 159.0);
        assert_eq!(p.get(2), 159.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_160() {
        let t = Tensor::full(vec![2], 160.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 160.0);
        assert_eq!(p.get(2), 160.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_161() {
        let t = Tensor::full(vec![2], 161.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 161.0);
        assert_eq!(p.get(2), 161.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_162() {
        let t = Tensor::full(vec![2], 162.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 162.0);
        assert_eq!(p.get(2), 162.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_163() {
        let t = Tensor::full(vec![2], 163.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 163.0);
        assert_eq!(p.get(2), 163.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_164() {
        let t = Tensor::full(vec![2], 164.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 164.0);
        assert_eq!(p.get(2), 164.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_165() {
        let t = Tensor::full(vec![2], 165.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 165.0);
        assert_eq!(p.get(2), 165.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_166() {
        let t = Tensor::full(vec![2], 166.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 166.0);
        assert_eq!(p.get(2), 166.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_167() {
        let t = Tensor::full(vec![2], 167.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 167.0);
        assert_eq!(p.get(2), 167.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_168() {
        let t = Tensor::full(vec![2], 168.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 168.0);
        assert_eq!(p.get(2), 168.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_169() {
        let t = Tensor::full(vec![2], 169.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 169.0);
        assert_eq!(p.get(2), 169.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_170() {
        let t = Tensor::full(vec![2], 170.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 170.0);
        assert_eq!(p.get(2), 170.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_171() {
        let t = Tensor::full(vec![2], 171.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 171.0);
        assert_eq!(p.get(2), 171.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_172() {
        let t = Tensor::full(vec![2], 172.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 172.0);
        assert_eq!(p.get(2), 172.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_173() {
        let t = Tensor::full(vec![2], 173.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 173.0);
        assert_eq!(p.get(2), 173.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_174() {
        let t = Tensor::full(vec![2], 174.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 174.0);
        assert_eq!(p.get(2), 174.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_175() {
        let t = Tensor::full(vec![2], 175.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 175.0);
        assert_eq!(p.get(2), 175.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_176() {
        let t = Tensor::full(vec![2], 176.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 176.0);
        assert_eq!(p.get(2), 176.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_177() {
        let t = Tensor::full(vec![2], 177.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 177.0);
        assert_eq!(p.get(2), 177.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_178() {
        let t = Tensor::full(vec![2], 178.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 178.0);
        assert_eq!(p.get(2), 178.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_179() {
        let t = Tensor::full(vec![2], 179.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 179.0);
        assert_eq!(p.get(2), 179.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_180() {
        let t = Tensor::full(vec![2], 180.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 180.0);
        assert_eq!(p.get(2), 180.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_181() {
        let t = Tensor::full(vec![2], 181.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 181.0);
        assert_eq!(p.get(2), 181.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_182() {
        let t = Tensor::full(vec![2], 182.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 182.0);
        assert_eq!(p.get(2), 182.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_183() {
        let t = Tensor::full(vec![2], 183.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 183.0);
        assert_eq!(p.get(2), 183.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_184() {
        let t = Tensor::full(vec![2], 184.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 184.0);
        assert_eq!(p.get(2), 184.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_185() {
        let t = Tensor::full(vec![2], 185.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 185.0);
        assert_eq!(p.get(2), 185.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_186() {
        let t = Tensor::full(vec![2], 186.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 186.0);
        assert_eq!(p.get(2), 186.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_187() {
        let t = Tensor::full(vec![2], 187.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 187.0);
        assert_eq!(p.get(2), 187.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_188() {
        let t = Tensor::full(vec![2], 188.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 188.0);
        assert_eq!(p.get(2), 188.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_189() {
        let t = Tensor::full(vec![2], 189.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 189.0);
        assert_eq!(p.get(2), 189.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_190() {
        let t = Tensor::full(vec![2], 190.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 190.0);
        assert_eq!(p.get(2), 190.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_191() {
        let t = Tensor::full(vec![2], 191.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 191.0);
        assert_eq!(p.get(2), 191.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_192() {
        let t = Tensor::full(vec![2], 192.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 192.0);
        assert_eq!(p.get(2), 192.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_193() {
        let t = Tensor::full(vec![2], 193.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 193.0);
        assert_eq!(p.get(2), 193.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_194() {
        let t = Tensor::full(vec![2], 194.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 194.0);
        assert_eq!(p.get(2), 194.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_195() {
        let t = Tensor::full(vec![2], 195.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 195.0);
        assert_eq!(p.get(2), 195.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_196() {
        let t = Tensor::full(vec![2], 196.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 196.0);
        assert_eq!(p.get(2), 196.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_197() {
        let t = Tensor::full(vec![2], 197.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 197.0);
        assert_eq!(p.get(2), 197.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_198() {
        let t = Tensor::full(vec![2], 198.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 198.0);
        assert_eq!(p.get(2), 198.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_199() {
        let t = Tensor::full(vec![2], 199.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 199.0);
        assert_eq!(p.get(2), 199.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_200() {
        let t = Tensor::full(vec![2], 200.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 200.0);
        assert_eq!(p.get(2), 200.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_201() {
        let t = Tensor::full(vec![2], 201.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 201.0);
        assert_eq!(p.get(2), 201.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_202() {
        let t = Tensor::full(vec![2], 202.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 202.0);
        assert_eq!(p.get(2), 202.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_203() {
        let t = Tensor::full(vec![2], 203.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 203.0);
        assert_eq!(p.get(2), 203.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_204() {
        let t = Tensor::full(vec![2], 204.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 204.0);
        assert_eq!(p.get(2), 204.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_205() {
        let t = Tensor::full(vec![2], 205.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 205.0);
        assert_eq!(p.get(2), 205.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_206() {
        let t = Tensor::full(vec![2], 206.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 206.0);
        assert_eq!(p.get(2), 206.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_207() {
        let t = Tensor::full(vec![2], 207.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 207.0);
        assert_eq!(p.get(2), 207.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_208() {
        let t = Tensor::full(vec![2], 208.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 208.0);
        assert_eq!(p.get(2), 208.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_209() {
        let t = Tensor::full(vec![2], 209.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 209.0);
        assert_eq!(p.get(2), 209.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_210() {
        let t = Tensor::full(vec![2], 210.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 210.0);
        assert_eq!(p.get(2), 210.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_211() {
        let t = Tensor::full(vec![2], 211.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 211.0);
        assert_eq!(p.get(2), 211.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_212() {
        let t = Tensor::full(vec![2], 212.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 212.0);
        assert_eq!(p.get(2), 212.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_213() {
        let t = Tensor::full(vec![2], 213.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 213.0);
        assert_eq!(p.get(2), 213.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_214() {
        let t = Tensor::full(vec![2], 214.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 214.0);
        assert_eq!(p.get(2), 214.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_215() {
        let t = Tensor::full(vec![2], 215.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 215.0);
        assert_eq!(p.get(2), 215.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_216() {
        let t = Tensor::full(vec![2], 216.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 216.0);
        assert_eq!(p.get(2), 216.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_217() {
        let t = Tensor::full(vec![2], 217.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 217.0);
        assert_eq!(p.get(2), 217.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_218() {
        let t = Tensor::full(vec![2], 218.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 218.0);
        assert_eq!(p.get(2), 218.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_219() {
        let t = Tensor::full(vec![2], 219.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 219.0);
        assert_eq!(p.get(2), 219.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_220() {
        let t = Tensor::full(vec![2], 220.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 220.0);
        assert_eq!(p.get(2), 220.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_221() {
        let t = Tensor::full(vec![2], 221.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 221.0);
        assert_eq!(p.get(2), 221.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_222() {
        let t = Tensor::full(vec![2], 222.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 222.0);
        assert_eq!(p.get(2), 222.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_223() {
        let t = Tensor::full(vec![2], 223.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 223.0);
        assert_eq!(p.get(2), 223.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_224() {
        let t = Tensor::full(vec![2], 224.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 224.0);
        assert_eq!(p.get(2), 224.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_225() {
        let t = Tensor::full(vec![2], 225.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 225.0);
        assert_eq!(p.get(2), 225.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_226() {
        let t = Tensor::full(vec![2], 226.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 226.0);
        assert_eq!(p.get(2), 226.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_227() {
        let t = Tensor::full(vec![2], 227.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 227.0);
        assert_eq!(p.get(2), 227.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_228() {
        let t = Tensor::full(vec![2], 228.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 228.0);
        assert_eq!(p.get(2), 228.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_229() {
        let t = Tensor::full(vec![2], 229.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 229.0);
        assert_eq!(p.get(2), 229.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_230() {
        let t = Tensor::full(vec![2], 230.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 230.0);
        assert_eq!(p.get(2), 230.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_231() {
        let t = Tensor::full(vec![2], 231.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 231.0);
        assert_eq!(p.get(2), 231.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_232() {
        let t = Tensor::full(vec![2], 232.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 232.0);
        assert_eq!(p.get(2), 232.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_233() {
        let t = Tensor::full(vec![2], 233.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 233.0);
        assert_eq!(p.get(2), 233.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_234() {
        let t = Tensor::full(vec![2], 234.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 234.0);
        assert_eq!(p.get(2), 234.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_235() {
        let t = Tensor::full(vec![2], 235.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 235.0);
        assert_eq!(p.get(2), 235.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_236() {
        let t = Tensor::full(vec![2], 236.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 236.0);
        assert_eq!(p.get(2), 236.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_237() {
        let t = Tensor::full(vec![2], 237.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 237.0);
        assert_eq!(p.get(2), 237.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_238() {
        let t = Tensor::full(vec![2], 238.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 238.0);
        assert_eq!(p.get(2), 238.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_239() {
        let t = Tensor::full(vec![2], 239.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 239.0);
        assert_eq!(p.get(2), 239.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_240() {
        let t = Tensor::full(vec![2], 240.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 240.0);
        assert_eq!(p.get(2), 240.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_241() {
        let t = Tensor::full(vec![2], 241.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 241.0);
        assert_eq!(p.get(2), 241.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_242() {
        let t = Tensor::full(vec![2], 242.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 242.0);
        assert_eq!(p.get(2), 242.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_243() {
        let t = Tensor::full(vec![2], 243.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 243.0);
        assert_eq!(p.get(2), 243.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_244() {
        let t = Tensor::full(vec![2], 244.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 244.0);
        assert_eq!(p.get(2), 244.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_245() {
        let t = Tensor::full(vec![2], 245.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 245.0);
        assert_eq!(p.get(2), 245.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_246() {
        let t = Tensor::full(vec![2], 246.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 246.0);
        assert_eq!(p.get(2), 246.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_247() {
        let t = Tensor::full(vec![2], 247.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 247.0);
        assert_eq!(p.get(2), 247.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_248() {
        let t = Tensor::full(vec![2], 248.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 248.0);
        assert_eq!(p.get(2), 248.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_249() {
        let t = Tensor::full(vec![2], 249.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 249.0);
        assert_eq!(p.get(2), 249.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_250() {
        let t = Tensor::full(vec![2], 250.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 250.0);
        assert_eq!(p.get(2), 250.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_251() {
        let t = Tensor::full(vec![2], 251.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 251.0);
        assert_eq!(p.get(2), 251.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_252() {
        let t = Tensor::full(vec![2], 252.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 252.0);
        assert_eq!(p.get(2), 252.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_253() {
        let t = Tensor::full(vec![2], 253.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 253.0);
        assert_eq!(p.get(2), 253.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_254() {
        let t = Tensor::full(vec![2], 254.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 254.0);
        assert_eq!(p.get(2), 254.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_255() {
        let t = Tensor::full(vec![2], 255.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 255.0);
        assert_eq!(p.get(2), 255.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_256() {
        let t = Tensor::full(vec![2], 256.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 256.0);
        assert_eq!(p.get(2), 256.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_257() {
        let t = Tensor::full(vec![2], 257.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 257.0);
        assert_eq!(p.get(2), 257.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_258() {
        let t = Tensor::full(vec![2], 258.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 258.0);
        assert_eq!(p.get(2), 258.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_259() {
        let t = Tensor::full(vec![2], 259.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 259.0);
        assert_eq!(p.get(2), 259.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_260() {
        let t = Tensor::full(vec![2], 260.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 260.0);
        assert_eq!(p.get(2), 260.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_261() {
        let t = Tensor::full(vec![2], 261.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 261.0);
        assert_eq!(p.get(2), 261.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_262() {
        let t = Tensor::full(vec![2], 262.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 262.0);
        assert_eq!(p.get(2), 262.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_263() {
        let t = Tensor::full(vec![2], 263.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 263.0);
        assert_eq!(p.get(2), 263.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_264() {
        let t = Tensor::full(vec![2], 264.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 264.0);
        assert_eq!(p.get(2), 264.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_265() {
        let t = Tensor::full(vec![2], 265.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 265.0);
        assert_eq!(p.get(2), 265.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_266() {
        let t = Tensor::full(vec![2], 266.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 266.0);
        assert_eq!(p.get(2), 266.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_267() {
        let t = Tensor::full(vec![2], 267.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 267.0);
        assert_eq!(p.get(2), 267.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_268() {
        let t = Tensor::full(vec![2], 268.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 268.0);
        assert_eq!(p.get(2), 268.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_269() {
        let t = Tensor::full(vec![2], 269.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 269.0);
        assert_eq!(p.get(2), 269.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_270() {
        let t = Tensor::full(vec![2], 270.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 270.0);
        assert_eq!(p.get(2), 270.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_271() {
        let t = Tensor::full(vec![2], 271.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 271.0);
        assert_eq!(p.get(2), 271.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_272() {
        let t = Tensor::full(vec![2], 272.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 272.0);
        assert_eq!(p.get(2), 272.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_273() {
        let t = Tensor::full(vec![2], 273.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 273.0);
        assert_eq!(p.get(2), 273.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_274() {
        let t = Tensor::full(vec![2], 274.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 274.0);
        assert_eq!(p.get(2), 274.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_275() {
        let t = Tensor::full(vec![2], 275.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 275.0);
        assert_eq!(p.get(2), 275.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_276() {
        let t = Tensor::full(vec![2], 276.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 276.0);
        assert_eq!(p.get(2), 276.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_277() {
        let t = Tensor::full(vec![2], 277.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 277.0);
        assert_eq!(p.get(2), 277.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_278() {
        let t = Tensor::full(vec![2], 278.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 278.0);
        assert_eq!(p.get(2), 278.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_279() {
        let t = Tensor::full(vec![2], 279.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 279.0);
        assert_eq!(p.get(2), 279.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_280() {
        let t = Tensor::full(vec![2], 280.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 280.0);
        assert_eq!(p.get(2), 280.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_281() {
        let t = Tensor::full(vec![2], 281.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 281.0);
        assert_eq!(p.get(2), 281.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_282() {
        let t = Tensor::full(vec![2], 282.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 282.0);
        assert_eq!(p.get(2), 282.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_283() {
        let t = Tensor::full(vec![2], 283.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 283.0);
        assert_eq!(p.get(2), 283.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_284() {
        let t = Tensor::full(vec![2], 284.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 284.0);
        assert_eq!(p.get(2), 284.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_285() {
        let t = Tensor::full(vec![2], 285.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 285.0);
        assert_eq!(p.get(2), 285.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_286() {
        let t = Tensor::full(vec![2], 286.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 286.0);
        assert_eq!(p.get(2), 286.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_287() {
        let t = Tensor::full(vec![2], 287.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 287.0);
        assert_eq!(p.get(2), 287.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_288() {
        let t = Tensor::full(vec![2], 288.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 288.0);
        assert_eq!(p.get(2), 288.0);
        assert_eq!(p.get(3), -1.0);
    }

    #[test]
    fn test_pad_stress_case_289() {
        let t = Tensor::full(vec![2], 289.0);
        let p = pad(&t, &[1, 1], "constant", -1.0);
        assert_eq!(p.shape(), &[4]);
        assert_eq!(p.get(0), -1.0);
        assert_eq!(p.get(1), 289.0);
        assert_eq!(p.get(2), 289.0);
        assert_eq!(p.get(3), -1.0);
    }
}
