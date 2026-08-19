//! Unfolding and folding operators (im2col, col2im, sliding window views).
//!
//! This module provides sliding window extractions and patch folding for image processing.

use crate::tensor::Tensor;

/// Returns a tensor containing all slices of size `size` along `dimension` with stride `step`.
pub fn unfold(input: &Tensor, dimension: usize, size: usize, step: usize) -> Tensor {
    assert!(dimension < input.ndim());
    assert!(size > 0 && step > 0);
    let dim_len = input.shape()[dimension];
    assert!(dim_len >= size, "unfold size exceeds dimension length");

    let num_slices = (dim_len - size) / step + 1;
    let mut new_shape = input.shape().to_vec();
    new_shape[dimension] = num_slices;
    new_shape.push(size);

    let numel: usize = new_shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let rank = new_shape.len();
    let mut coords = vec![0usize; rank];

    for _ in 0..numel {
        let mut src_coords = coords[..input.ndim()].to_vec();
        let slice_idx = coords[dimension];
        let offset_in_slice = coords[rank - 1];
        src_coords[dimension] = slice_idx * step + offset_in_slice;

        out.push(input.get_index(&src_coords));

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

/// Combines an array of sliding local blocks into a large output image (col2im).
pub fn fold(
    input: &Tensor,
    output_size: (usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    let (out_h, out_w) = output_size;
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let n = input.shape()[0];
    let in_c = input.shape()[1] / (kh * kw);
    let num_patches = input.shape()[2];

    let mut out = Tensor::zeros(vec![n, in_c, out_h, out_w]);
    let patches_w = (out_w + 2 * pw - kw) / sw + 1;

    for b in 0..n {
        for p in 0..num_patches {
            let ph_idx = p / patches_w;
            let pw_idx = p % patches_w;
            let h_start = (ph_idx * sh) as isize - ph as isize;
            let w_start = (pw_idx * sw) as isize - pw as isize;

            let mut patch_elem = 0;
            for c in 0..in_c {
                for f_h in 0..kh {
                    let ih = h_start + f_h as isize;
                    for f_w in 0..kw {
                        let iw = w_start + f_w as isize;
                        let val = input.get_3d(b, patch_elem, p);
                        patch_elem += 1;

                        if ih >= 0 && (ih as usize) < out_h && iw >= 0 && (iw as usize) < out_w {
                            let cur = out.get_4d(b, c, ih as usize, iw as usize);
                            out.set_4d(b, c, ih as usize, iw as usize, cur + val);
                        }
                    }
                }
            }
        }
    }

    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unfold_1d() {
        let t = Tensor::arange(0.0, 5.0, 1.0);
        let u = unfold(&t, 0, 3, 1);
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(u.get_2d(0, 0), 0.0);
        assert_eq!(u.get_2d(0, 2), 2.0);
        assert_eq!(u.get_2d(2, 0), 2.0);
    }

    #[test]
    fn test_unfold_operation() {
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let u = unfold(&input, 1, 2, 1);
        assert_eq!(u.shape(), &[2, 2, 2]);
    }
}
