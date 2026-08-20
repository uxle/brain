//! Multi-dimensional spatial pooling operators (Max, Average, Adaptive, Global).
//!
//! This module provides 2D/3D spatial pooling layers for convolutional neural network architectures.

use crate::tensor::Tensor;

// =============================================================================
// Max & Average Pooling 2D
// =============================================================================

/// 2D Max Pooling on (N, C, H, W) tensors.
pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    assert_eq!(input.ndim(), 4, "max_pool2d requires 4D tensor");
    let (n, c, in_h, in_w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    // Checked i128 dim computation: a kernel larger than the padded input
    // yields an empty output instead of an usize underflow -> OOM/panic.
    let out_h = crate::shape::Shape::output_dim(in_h, ph, kh, sh, 1);
    let out_w = crate::shape::Shape::output_dim(in_w, pw, kw, sw, 1);
    let mut out = Tensor::zeros(vec![n, c, out_h, out_w]);

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f64::NEG_INFINITY;
                    let h_start = (oh * sh) as isize - ph as isize;
                    let w_start = (ow * sw) as isize - pw as isize;

                    for f_h in 0..kh {
                        let ih = h_start + f_h as isize;
                        if ih >= 0 && (ih as usize) < in_h {
                            for f_w in 0..kw {
                                let iw = w_start + f_w as isize;
                                if iw >= 0 && (iw as usize) < in_w {
                                    let v = input.get_4d(b, ch, ih as usize, iw as usize);
                                    if v > max_val {
                                        max_val = v;
                                    }
                                }
                            }
                        }
                    }
                    out.set_4d(b, ch, oh, ow, max_val);
                }
            }
        }
    }
    out
}

/// 2D Average Pooling on (N, C, H, W) tensors.
pub fn avg_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    assert_eq!(input.ndim(), 4, "avg_pool2d requires 4D tensor");
    let (n, c, in_h, in_w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let out_h = crate::shape::Shape::output_dim(in_h, ph, kh, sh, 1);
    let out_w = crate::shape::Shape::output_dim(in_w, pw, kw, sw, 1);
    let mut out = Tensor::zeros(vec![n, c, out_h, out_w]);
    let window_area = (kh * kw) as f64;

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;
                    let h_start = (oh * sh) as isize - ph as isize;
                    let w_start = (ow * sw) as isize - pw as isize;

                    for f_h in 0..kh {
                        let ih = h_start + f_h as isize;
                        if ih >= 0 && (ih as usize) < in_h {
                            for f_w in 0..kw {
                                let iw = w_start + f_w as isize;
                                if iw >= 0 && (iw as usize) < in_w {
                                    sum += input.get_4d(b, ch, ih as usize, iw as usize);
                                }
                            }
                        }
                    }
                    out.set_4d(b, ch, oh, ow, sum / window_area);
                }
            }
        }
    }
    out
}

/// Global Average Pooling 2D reducing spatial dimensions to 1x1.
pub fn global_avg_pool2d(input: &Tensor) -> Tensor {
    assert_eq!(input.ndim(), 4);
    let (in_h, in_w) = (input.shape()[2], input.shape()[3]);
    avg_pool2d(input, (in_h, in_w), (1, 1), (0, 0))
}

/// Global Max Pooling 2D reducing spatial dimensions to 1x1.
pub fn global_max_pool2d(input: &Tensor) -> Tensor {
    assert_eq!(input.ndim(), 4);
    let (in_h, in_w) = (input.shape()[2], input.shape()[3]);
    max_pool2d(input, (in_h, in_w), (1, 1), (0, 0))
}

// =============================================================================
// Adaptive Pooling 2D
// =============================================================================

/// Adaptive Average Pooling 2D to a fixed output size (torch semantics).
pub fn adaptive_avg_pool2d(input: &Tensor, out_h: usize, out_w: usize) -> Tensor {
    assert_eq!(input.ndim(), 4, "adaptive_avg_pool2d requires 4D tensor");
    assert!(
        out_h > 0 && out_w > 0,
        "adaptive pooling output size must be positive"
    );
    let (n, c, in_h, in_w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let mut out = Tensor::zeros(vec![n, c, out_h, out_w]);

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                let h_start = (oh * in_h) / out_h;
                let h_end = (((oh + 1) * in_h) + out_h - 1) / out_h;
                for ow in 0..out_w {
                    let w_start = (ow * in_w) / out_w;
                    let w_end = (((ow + 1) * in_w) + out_w - 1) / out_w;
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += input.get_4d(b, ch, ih, iw);
                            count += 1;
                        }
                    }
                    out.set_4d(
                        b,
                        ch,
                        oh,
                        ow,
                        if count > 0 { sum / (count as f64) } else { 0.0 },
                    );
                }
            }
        }
    }
    out
}

/// Adaptive Max Pooling 2D to a fixed output size (torch semantics).
pub fn adaptive_max_pool2d(input: &Tensor, out_h: usize, out_w: usize) -> Tensor {
    assert_eq!(input.ndim(), 4, "adaptive_max_pool2d requires 4D tensor");
    assert!(
        out_h > 0 && out_w > 0,
        "adaptive pooling output size must be positive"
    );
    let (n, c, in_h, in_w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let mut out = Tensor::zeros(vec![n, c, out_h, out_w]);

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                let h_start = (oh * in_h) / out_h;
                let h_end = (((oh + 1) * in_h) + out_h - 1) / out_h;
                for ow in 0..out_w {
                    let w_start = (ow * in_w) / out_w;
                    let w_end = (((ow + 1) * in_w) + out_w - 1) / out_w;
                    let mut max_val = f64::NEG_INFINITY;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let v = input.get_4d(b, ch, ih, iw);
                            if v > max_val {
                                max_val = v;
                            }
                        }
                    }
                    out.set_4d(b, ch, oh, ow, max_val);
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
    fn test_max_pool2d() {
        let input = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0], vec![1, 1, 2, 2]);
        let out = max_pool2d(&input, (2, 2), (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 4.0);
    }

    #[test]
    fn test_avg_pool2d() {
        let input = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0], vec![1, 1, 2, 2]);
        let out = avg_pool2d(&input, (2, 2), (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 2.5);
    }

    #[test]
    fn test_pooling_spatial_downsampling() {
        let input = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![1, 1, 4, 4],
        );

        let max_p = max_pool2d(&input, (2, 2), (2, 2), (0, 0));
        assert_eq!(max_p.shape(), &[1, 1, 2, 2]);
        assert_eq!(max_p.to_vec(), vec![6.0, 8.0, 14.0, 16.0]);

        let avg_p = avg_pool2d(&input, (2, 2), (2, 2), (0, 0));
        assert_eq!(avg_p.shape(), &[1, 1, 2, 2]);
        assert_eq!(avg_p.to_vec(), vec![3.5, 5.5, 11.5, 13.5]);

        let g_avg = global_avg_pool2d(&input);
        assert_eq!(g_avg.shape(), &[1, 1, 1, 1]);
        assert_eq!(g_avg.to_vec(), vec![8.5]);
    }

    #[test]
    fn test_adaptive_pool2d() {
        let input = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![1, 1, 4, 4],
        );

        // Upsample to 6x6 via adaptive avg: overlaps produce repeats of pooled regions
        let up = adaptive_avg_pool2d(&input, 6, 6);
        assert_eq!(up.shape(), &[1, 1, 6, 6]);

        // Downsample 4x4 -> 2x2: same as regular 2x2 pooling
        let avg = adaptive_avg_pool2d(&input, 2, 2);
        assert_eq!(avg.to_vec(), vec![3.5, 5.5, 11.5, 13.5]);
        let max = adaptive_max_pool2d(&input, 2, 2);
        assert_eq!(max.to_vec(), vec![6.0, 8.0, 14.0, 16.0]);

        // Reduce to 1x1: global stats
        let g = adaptive_avg_pool2d(&input, 1, 1);
        assert_eq!(g.to_vec(), vec![8.5]);

        // Single-row region semantics: 4x4 -> 2x1 averages 2 rows x 4 cols
        let col = adaptive_avg_pool2d(&input, 2, 1);
        assert_eq!(col.shape(), &[1, 1, 2, 1]);
        assert!((col.get(0) - 4.5).abs() < 1e-9);
        assert!((col.get(1) - 12.5).abs() < 1e-9);
    }
}
