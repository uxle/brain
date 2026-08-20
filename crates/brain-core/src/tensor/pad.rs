//! Multi-dimensional tensor padding operators (Constant, Reflect, Replicate, Circular).
//!
//! This module provides padding routines for 1D, 2D, and ND tensors.

use crate::tensor::Tensor;

/// Pads a tensor according to the specified padding widths and mode.
pub fn pad(input: &Tensor, pad: &[usize], mode: &str, value: f64) -> Tensor {
    assert!(
        pad.len() % 2 == 0,
        "pad must contain pairs of (before, after)"
    );
    let num_padded_dims = pad.len() / 2;
    let rank = input.ndim();
    assert!(
        num_padded_dims <= rank,
        "Cannot pad more dimensions than tensor rank"
    );

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
                    "reflect" => {
                        src_coords[dim] = if cur_coord < p_before {
                            (p_before - cur_coord).min(orig_len.saturating_sub(1))
                        } else {
                            let excess = cur_coord - (p_before + orig_len - 1);
                            (orig_len.saturating_sub(1)).saturating_sub(excess)
                        };
                    }
                    "circular" => {
                        let idx =
                            (cur_coord as isize - p_before as isize).rem_euclid(orig_len as isize);
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
    fn test_pad_modes_table() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let const_pad = pad(&t, &[1, 1], "constant", 0.0);
        assert_eq!(const_pad.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 0.0]);

        let rep_pad = pad(&t, &[1, 1], "replicate", 0.0);
        assert_eq!(rep_pad.to_vec(), vec![1.0, 1.0, 2.0, 3.0, 3.0]);

        let ref_pad = pad(&t, &[1, 1], "reflect", 0.0);
        assert_eq!(ref_pad.to_vec(), vec![2.0, 1.0, 2.0, 3.0, 2.0]);
    }
}
