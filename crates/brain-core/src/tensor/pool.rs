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

    let out_h = (in_h + 2 * ph - kh) / sh + 1;
    let out_w = (in_w + 2 * pw - kw) / sw + 1;
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

    let out_h = (in_h + 2 * ph - kh) / sh + 1;
    let out_w = (in_w + 2 * pw - kw) / sw + 1;
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
    fn test_pool_stress_case_001() {
        let input = Tensor::full(vec![1, 1, 4, 4], 1.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 1.0);
    }

    #[test]
    fn test_pool_stress_case_002() {
        let input = Tensor::full(vec![1, 1, 4, 4], 2.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 2.0);
    }

    #[test]
    fn test_pool_stress_case_003() {
        let input = Tensor::full(vec![1, 1, 4, 4], 3.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 3.0);
    }

    #[test]
    fn test_pool_stress_case_004() {
        let input = Tensor::full(vec![1, 1, 4, 4], 4.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 4.0);
    }

    #[test]
    fn test_pool_stress_case_005() {
        let input = Tensor::full(vec![1, 1, 4, 4], 5.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 5.0);
    }

    #[test]
    fn test_pool_stress_case_006() {
        let input = Tensor::full(vec![1, 1, 4, 4], 6.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 6.0);
    }

    #[test]
    fn test_pool_stress_case_007() {
        let input = Tensor::full(vec![1, 1, 4, 4], 7.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 7.0);
    }

    #[test]
    fn test_pool_stress_case_008() {
        let input = Tensor::full(vec![1, 1, 4, 4], 8.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 8.0);
    }

    #[test]
    fn test_pool_stress_case_009() {
        let input = Tensor::full(vec![1, 1, 4, 4], 9.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 9.0);
    }

    #[test]
    fn test_pool_stress_case_010() {
        let input = Tensor::full(vec![1, 1, 4, 4], 10.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 10.0);
    }

    #[test]
    fn test_pool_stress_case_011() {
        let input = Tensor::full(vec![1, 1, 4, 4], 11.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 11.0);
    }

    #[test]
    fn test_pool_stress_case_012() {
        let input = Tensor::full(vec![1, 1, 4, 4], 12.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 12.0);
    }

    #[test]
    fn test_pool_stress_case_013() {
        let input = Tensor::full(vec![1, 1, 4, 4], 13.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 13.0);
    }

    #[test]
    fn test_pool_stress_case_014() {
        let input = Tensor::full(vec![1, 1, 4, 4], 14.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 14.0);
    }

    #[test]
    fn test_pool_stress_case_015() {
        let input = Tensor::full(vec![1, 1, 4, 4], 15.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 15.0);
    }

    #[test]
    fn test_pool_stress_case_016() {
        let input = Tensor::full(vec![1, 1, 4, 4], 16.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 16.0);
    }

    #[test]
    fn test_pool_stress_case_017() {
        let input = Tensor::full(vec![1, 1, 4, 4], 17.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 17.0);
    }

    #[test]
    fn test_pool_stress_case_018() {
        let input = Tensor::full(vec![1, 1, 4, 4], 18.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 18.0);
    }

    #[test]
    fn test_pool_stress_case_019() {
        let input = Tensor::full(vec![1, 1, 4, 4], 19.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 19.0);
    }

    #[test]
    fn test_pool_stress_case_020() {
        let input = Tensor::full(vec![1, 1, 4, 4], 20.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 20.0);
    }

    #[test]
    fn test_pool_stress_case_021() {
        let input = Tensor::full(vec![1, 1, 4, 4], 21.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 21.0);
    }

    #[test]
    fn test_pool_stress_case_022() {
        let input = Tensor::full(vec![1, 1, 4, 4], 22.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 22.0);
    }

    #[test]
    fn test_pool_stress_case_023() {
        let input = Tensor::full(vec![1, 1, 4, 4], 23.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 23.0);
    }

    #[test]
    fn test_pool_stress_case_024() {
        let input = Tensor::full(vec![1, 1, 4, 4], 24.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 24.0);
    }

    #[test]
    fn test_pool_stress_case_025() {
        let input = Tensor::full(vec![1, 1, 4, 4], 25.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 25.0);
    }

    #[test]
    fn test_pool_stress_case_026() {
        let input = Tensor::full(vec![1, 1, 4, 4], 26.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 26.0);
    }

    #[test]
    fn test_pool_stress_case_027() {
        let input = Tensor::full(vec![1, 1, 4, 4], 27.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 27.0);
    }

    #[test]
    fn test_pool_stress_case_028() {
        let input = Tensor::full(vec![1, 1, 4, 4], 28.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 28.0);
    }

    #[test]
    fn test_pool_stress_case_029() {
        let input = Tensor::full(vec![1, 1, 4, 4], 29.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 29.0);
    }

    #[test]
    fn test_pool_stress_case_030() {
        let input = Tensor::full(vec![1, 1, 4, 4], 30.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 30.0);
    }

    #[test]
    fn test_pool_stress_case_031() {
        let input = Tensor::full(vec![1, 1, 4, 4], 31.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 31.0);
    }

    #[test]
    fn test_pool_stress_case_032() {
        let input = Tensor::full(vec![1, 1, 4, 4], 32.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 32.0);
    }

    #[test]
    fn test_pool_stress_case_033() {
        let input = Tensor::full(vec![1, 1, 4, 4], 33.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 33.0);
    }

    #[test]
    fn test_pool_stress_case_034() {
        let input = Tensor::full(vec![1, 1, 4, 4], 34.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 34.0);
    }

    #[test]
    fn test_pool_stress_case_035() {
        let input = Tensor::full(vec![1, 1, 4, 4], 35.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 35.0);
    }

    #[test]
    fn test_pool_stress_case_036() {
        let input = Tensor::full(vec![1, 1, 4, 4], 36.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 36.0);
    }

    #[test]
    fn test_pool_stress_case_037() {
        let input = Tensor::full(vec![1, 1, 4, 4], 37.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 37.0);
    }

    #[test]
    fn test_pool_stress_case_038() {
        let input = Tensor::full(vec![1, 1, 4, 4], 38.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 38.0);
    }

    #[test]
    fn test_pool_stress_case_039() {
        let input = Tensor::full(vec![1, 1, 4, 4], 39.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 39.0);
    }

    #[test]
    fn test_pool_stress_case_040() {
        let input = Tensor::full(vec![1, 1, 4, 4], 40.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 40.0);
    }

    #[test]
    fn test_pool_stress_case_041() {
        let input = Tensor::full(vec![1, 1, 4, 4], 41.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 41.0);
    }

    #[test]
    fn test_pool_stress_case_042() {
        let input = Tensor::full(vec![1, 1, 4, 4], 42.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 42.0);
    }

    #[test]
    fn test_pool_stress_case_043() {
        let input = Tensor::full(vec![1, 1, 4, 4], 43.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 43.0);
    }

    #[test]
    fn test_pool_stress_case_044() {
        let input = Tensor::full(vec![1, 1, 4, 4], 44.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 44.0);
    }

    #[test]
    fn test_pool_stress_case_045() {
        let input = Tensor::full(vec![1, 1, 4, 4], 45.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 45.0);
    }

    #[test]
    fn test_pool_stress_case_046() {
        let input = Tensor::full(vec![1, 1, 4, 4], 46.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 46.0);
    }

    #[test]
    fn test_pool_stress_case_047() {
        let input = Tensor::full(vec![1, 1, 4, 4], 47.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 47.0);
    }

    #[test]
    fn test_pool_stress_case_048() {
        let input = Tensor::full(vec![1, 1, 4, 4], 48.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 48.0);
    }

    #[test]
    fn test_pool_stress_case_049() {
        let input = Tensor::full(vec![1, 1, 4, 4], 49.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 49.0);
    }

    #[test]
    fn test_pool_stress_case_050() {
        let input = Tensor::full(vec![1, 1, 4, 4], 50.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 50.0);
    }

    #[test]
    fn test_pool_stress_case_051() {
        let input = Tensor::full(vec![1, 1, 4, 4], 51.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 51.0);
    }

    #[test]
    fn test_pool_stress_case_052() {
        let input = Tensor::full(vec![1, 1, 4, 4], 52.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 52.0);
    }

    #[test]
    fn test_pool_stress_case_053() {
        let input = Tensor::full(vec![1, 1, 4, 4], 53.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 53.0);
    }

    #[test]
    fn test_pool_stress_case_054() {
        let input = Tensor::full(vec![1, 1, 4, 4], 54.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 54.0);
    }

    #[test]
    fn test_pool_stress_case_055() {
        let input = Tensor::full(vec![1, 1, 4, 4], 55.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 55.0);
    }

    #[test]
    fn test_pool_stress_case_056() {
        let input = Tensor::full(vec![1, 1, 4, 4], 56.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 56.0);
    }

    #[test]
    fn test_pool_stress_case_057() {
        let input = Tensor::full(vec![1, 1, 4, 4], 57.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 57.0);
    }

    #[test]
    fn test_pool_stress_case_058() {
        let input = Tensor::full(vec![1, 1, 4, 4], 58.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 58.0);
    }

    #[test]
    fn test_pool_stress_case_059() {
        let input = Tensor::full(vec![1, 1, 4, 4], 59.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 59.0);
    }

    #[test]
    fn test_pool_stress_case_060() {
        let input = Tensor::full(vec![1, 1, 4, 4], 60.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 60.0);
    }

    #[test]
    fn test_pool_stress_case_061() {
        let input = Tensor::full(vec![1, 1, 4, 4], 61.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 61.0);
    }

    #[test]
    fn test_pool_stress_case_062() {
        let input = Tensor::full(vec![1, 1, 4, 4], 62.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 62.0);
    }

    #[test]
    fn test_pool_stress_case_063() {
        let input = Tensor::full(vec![1, 1, 4, 4], 63.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 63.0);
    }

    #[test]
    fn test_pool_stress_case_064() {
        let input = Tensor::full(vec![1, 1, 4, 4], 64.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 64.0);
    }

    #[test]
    fn test_pool_stress_case_065() {
        let input = Tensor::full(vec![1, 1, 4, 4], 65.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 65.0);
    }

    #[test]
    fn test_pool_stress_case_066() {
        let input = Tensor::full(vec![1, 1, 4, 4], 66.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 66.0);
    }

    #[test]
    fn test_pool_stress_case_067() {
        let input = Tensor::full(vec![1, 1, 4, 4], 67.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 67.0);
    }

    #[test]
    fn test_pool_stress_case_068() {
        let input = Tensor::full(vec![1, 1, 4, 4], 68.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 68.0);
    }

    #[test]
    fn test_pool_stress_case_069() {
        let input = Tensor::full(vec![1, 1, 4, 4], 69.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 69.0);
    }

    #[test]
    fn test_pool_stress_case_070() {
        let input = Tensor::full(vec![1, 1, 4, 4], 70.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 70.0);
    }

    #[test]
    fn test_pool_stress_case_071() {
        let input = Tensor::full(vec![1, 1, 4, 4], 71.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 71.0);
    }

    #[test]
    fn test_pool_stress_case_072() {
        let input = Tensor::full(vec![1, 1, 4, 4], 72.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 72.0);
    }

    #[test]
    fn test_pool_stress_case_073() {
        let input = Tensor::full(vec![1, 1, 4, 4], 73.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 73.0);
    }

    #[test]
    fn test_pool_stress_case_074() {
        let input = Tensor::full(vec![1, 1, 4, 4], 74.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 74.0);
    }

    #[test]
    fn test_pool_stress_case_075() {
        let input = Tensor::full(vec![1, 1, 4, 4], 75.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 75.0);
    }

    #[test]
    fn test_pool_stress_case_076() {
        let input = Tensor::full(vec![1, 1, 4, 4], 76.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 76.0);
    }

    #[test]
    fn test_pool_stress_case_077() {
        let input = Tensor::full(vec![1, 1, 4, 4], 77.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 77.0);
    }

    #[test]
    fn test_pool_stress_case_078() {
        let input = Tensor::full(vec![1, 1, 4, 4], 78.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 78.0);
    }

    #[test]
    fn test_pool_stress_case_079() {
        let input = Tensor::full(vec![1, 1, 4, 4], 79.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 79.0);
    }

    #[test]
    fn test_pool_stress_case_080() {
        let input = Tensor::full(vec![1, 1, 4, 4], 80.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 80.0);
    }

    #[test]
    fn test_pool_stress_case_081() {
        let input = Tensor::full(vec![1, 1, 4, 4], 81.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 81.0);
    }

    #[test]
    fn test_pool_stress_case_082() {
        let input = Tensor::full(vec![1, 1, 4, 4], 82.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 82.0);
    }

    #[test]
    fn test_pool_stress_case_083() {
        let input = Tensor::full(vec![1, 1, 4, 4], 83.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 83.0);
    }

    #[test]
    fn test_pool_stress_case_084() {
        let input = Tensor::full(vec![1, 1, 4, 4], 84.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 84.0);
    }

    #[test]
    fn test_pool_stress_case_085() {
        let input = Tensor::full(vec![1, 1, 4, 4], 85.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 85.0);
    }

    #[test]
    fn test_pool_stress_case_086() {
        let input = Tensor::full(vec![1, 1, 4, 4], 86.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 86.0);
    }

    #[test]
    fn test_pool_stress_case_087() {
        let input = Tensor::full(vec![1, 1, 4, 4], 87.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 87.0);
    }

    #[test]
    fn test_pool_stress_case_088() {
        let input = Tensor::full(vec![1, 1, 4, 4], 88.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 88.0);
    }

    #[test]
    fn test_pool_stress_case_089() {
        let input = Tensor::full(vec![1, 1, 4, 4], 89.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 89.0);
    }

    #[test]
    fn test_pool_stress_case_090() {
        let input = Tensor::full(vec![1, 1, 4, 4], 90.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 90.0);
    }

    #[test]
    fn test_pool_stress_case_091() {
        let input = Tensor::full(vec![1, 1, 4, 4], 91.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 91.0);
    }

    #[test]
    fn test_pool_stress_case_092() {
        let input = Tensor::full(vec![1, 1, 4, 4], 92.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 92.0);
    }

    #[test]
    fn test_pool_stress_case_093() {
        let input = Tensor::full(vec![1, 1, 4, 4], 93.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 93.0);
    }

    #[test]
    fn test_pool_stress_case_094() {
        let input = Tensor::full(vec![1, 1, 4, 4], 94.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 94.0);
    }

    #[test]
    fn test_pool_stress_case_095() {
        let input = Tensor::full(vec![1, 1, 4, 4], 95.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 95.0);
    }

    #[test]
    fn test_pool_stress_case_096() {
        let input = Tensor::full(vec![1, 1, 4, 4], 96.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 96.0);
    }

    #[test]
    fn test_pool_stress_case_097() {
        let input = Tensor::full(vec![1, 1, 4, 4], 97.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 97.0);
    }

    #[test]
    fn test_pool_stress_case_098() {
        let input = Tensor::full(vec![1, 1, 4, 4], 98.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 98.0);
    }

    #[test]
    fn test_pool_stress_case_099() {
        let input = Tensor::full(vec![1, 1, 4, 4], 99.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 99.0);
    }

    #[test]
    fn test_pool_stress_case_100() {
        let input = Tensor::full(vec![1, 1, 4, 4], 100.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 100.0);
    }

    #[test]
    fn test_pool_stress_case_101() {
        let input = Tensor::full(vec![1, 1, 4, 4], 101.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 101.0);
    }

    #[test]
    fn test_pool_stress_case_102() {
        let input = Tensor::full(vec![1, 1, 4, 4], 102.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 102.0);
    }

    #[test]
    fn test_pool_stress_case_103() {
        let input = Tensor::full(vec![1, 1, 4, 4], 103.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 103.0);
    }

    #[test]
    fn test_pool_stress_case_104() {
        let input = Tensor::full(vec![1, 1, 4, 4], 104.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 104.0);
    }

    #[test]
    fn test_pool_stress_case_105() {
        let input = Tensor::full(vec![1, 1, 4, 4], 105.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 105.0);
    }

    #[test]
    fn test_pool_stress_case_106() {
        let input = Tensor::full(vec![1, 1, 4, 4], 106.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 106.0);
    }

    #[test]
    fn test_pool_stress_case_107() {
        let input = Tensor::full(vec![1, 1, 4, 4], 107.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 107.0);
    }

    #[test]
    fn test_pool_stress_case_108() {
        let input = Tensor::full(vec![1, 1, 4, 4], 108.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 108.0);
    }

    #[test]
    fn test_pool_stress_case_109() {
        let input = Tensor::full(vec![1, 1, 4, 4], 109.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 109.0);
    }

    #[test]
    fn test_pool_stress_case_110() {
        let input = Tensor::full(vec![1, 1, 4, 4], 110.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 110.0);
    }

    #[test]
    fn test_pool_stress_case_111() {
        let input = Tensor::full(vec![1, 1, 4, 4], 111.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 111.0);
    }

    #[test]
    fn test_pool_stress_case_112() {
        let input = Tensor::full(vec![1, 1, 4, 4], 112.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 112.0);
    }

    #[test]
    fn test_pool_stress_case_113() {
        let input = Tensor::full(vec![1, 1, 4, 4], 113.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 113.0);
    }

    #[test]
    fn test_pool_stress_case_114() {
        let input = Tensor::full(vec![1, 1, 4, 4], 114.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 114.0);
    }

    #[test]
    fn test_pool_stress_case_115() {
        let input = Tensor::full(vec![1, 1, 4, 4], 115.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 115.0);
    }

    #[test]
    fn test_pool_stress_case_116() {
        let input = Tensor::full(vec![1, 1, 4, 4], 116.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 116.0);
    }

    #[test]
    fn test_pool_stress_case_117() {
        let input = Tensor::full(vec![1, 1, 4, 4], 117.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 117.0);
    }

    #[test]
    fn test_pool_stress_case_118() {
        let input = Tensor::full(vec![1, 1, 4, 4], 118.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 118.0);
    }

    #[test]
    fn test_pool_stress_case_119() {
        let input = Tensor::full(vec![1, 1, 4, 4], 119.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 119.0);
    }

    #[test]
    fn test_pool_stress_case_120() {
        let input = Tensor::full(vec![1, 1, 4, 4], 120.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 120.0);
    }

    #[test]
    fn test_pool_stress_case_121() {
        let input = Tensor::full(vec![1, 1, 4, 4], 121.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 121.0);
    }

    #[test]
    fn test_pool_stress_case_122() {
        let input = Tensor::full(vec![1, 1, 4, 4], 122.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 122.0);
    }

    #[test]
    fn test_pool_stress_case_123() {
        let input = Tensor::full(vec![1, 1, 4, 4], 123.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 123.0);
    }

    #[test]
    fn test_pool_stress_case_124() {
        let input = Tensor::full(vec![1, 1, 4, 4], 124.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 124.0);
    }

    #[test]
    fn test_pool_stress_case_125() {
        let input = Tensor::full(vec![1, 1, 4, 4], 125.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 125.0);
    }

    #[test]
    fn test_pool_stress_case_126() {
        let input = Tensor::full(vec![1, 1, 4, 4], 126.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 126.0);
    }

    #[test]
    fn test_pool_stress_case_127() {
        let input = Tensor::full(vec![1, 1, 4, 4], 127.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 127.0);
    }

    #[test]
    fn test_pool_stress_case_128() {
        let input = Tensor::full(vec![1, 1, 4, 4], 128.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 128.0);
    }

    #[test]
    fn test_pool_stress_case_129() {
        let input = Tensor::full(vec![1, 1, 4, 4], 129.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 129.0);
    }

    #[test]
    fn test_pool_stress_case_130() {
        let input = Tensor::full(vec![1, 1, 4, 4], 130.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 130.0);
    }

    #[test]
    fn test_pool_stress_case_131() {
        let input = Tensor::full(vec![1, 1, 4, 4], 131.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 131.0);
    }

    #[test]
    fn test_pool_stress_case_132() {
        let input = Tensor::full(vec![1, 1, 4, 4], 132.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 132.0);
    }

    #[test]
    fn test_pool_stress_case_133() {
        let input = Tensor::full(vec![1, 1, 4, 4], 133.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 133.0);
    }

    #[test]
    fn test_pool_stress_case_134() {
        let input = Tensor::full(vec![1, 1, 4, 4], 134.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 134.0);
    }

    #[test]
    fn test_pool_stress_case_135() {
        let input = Tensor::full(vec![1, 1, 4, 4], 135.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 135.0);
    }

    #[test]
    fn test_pool_stress_case_136() {
        let input = Tensor::full(vec![1, 1, 4, 4], 136.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 136.0);
    }

    #[test]
    fn test_pool_stress_case_137() {
        let input = Tensor::full(vec![1, 1, 4, 4], 137.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 137.0);
    }

    #[test]
    fn test_pool_stress_case_138() {
        let input = Tensor::full(vec![1, 1, 4, 4], 138.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 138.0);
    }

    #[test]
    fn test_pool_stress_case_139() {
        let input = Tensor::full(vec![1, 1, 4, 4], 139.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 139.0);
    }

    #[test]
    fn test_pool_stress_case_140() {
        let input = Tensor::full(vec![1, 1, 4, 4], 140.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 140.0);
    }

    #[test]
    fn test_pool_stress_case_141() {
        let input = Tensor::full(vec![1, 1, 4, 4], 141.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 141.0);
    }

    #[test]
    fn test_pool_stress_case_142() {
        let input = Tensor::full(vec![1, 1, 4, 4], 142.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 142.0);
    }

    #[test]
    fn test_pool_stress_case_143() {
        let input = Tensor::full(vec![1, 1, 4, 4], 143.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 143.0);
    }

    #[test]
    fn test_pool_stress_case_144() {
        let input = Tensor::full(vec![1, 1, 4, 4], 144.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 144.0);
    }

    #[test]
    fn test_pool_stress_case_145() {
        let input = Tensor::full(vec![1, 1, 4, 4], 145.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 145.0);
    }

    #[test]
    fn test_pool_stress_case_146() {
        let input = Tensor::full(vec![1, 1, 4, 4], 146.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 146.0);
    }

    #[test]
    fn test_pool_stress_case_147() {
        let input = Tensor::full(vec![1, 1, 4, 4], 147.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 147.0);
    }

    #[test]
    fn test_pool_stress_case_148() {
        let input = Tensor::full(vec![1, 1, 4, 4], 148.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 148.0);
    }

    #[test]
    fn test_pool_stress_case_149() {
        let input = Tensor::full(vec![1, 1, 4, 4], 149.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 149.0);
    }

    #[test]
    fn test_pool_stress_case_150() {
        let input = Tensor::full(vec![1, 1, 4, 4], 150.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 150.0);
    }

    #[test]
    fn test_pool_stress_case_151() {
        let input = Tensor::full(vec![1, 1, 4, 4], 151.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 151.0);
    }

    #[test]
    fn test_pool_stress_case_152() {
        let input = Tensor::full(vec![1, 1, 4, 4], 152.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 152.0);
    }

    #[test]
    fn test_pool_stress_case_153() {
        let input = Tensor::full(vec![1, 1, 4, 4], 153.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 153.0);
    }

    #[test]
    fn test_pool_stress_case_154() {
        let input = Tensor::full(vec![1, 1, 4, 4], 154.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 154.0);
    }

    #[test]
    fn test_pool_stress_case_155() {
        let input = Tensor::full(vec![1, 1, 4, 4], 155.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 155.0);
    }

    #[test]
    fn test_pool_stress_case_156() {
        let input = Tensor::full(vec![1, 1, 4, 4], 156.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 156.0);
    }

    #[test]
    fn test_pool_stress_case_157() {
        let input = Tensor::full(vec![1, 1, 4, 4], 157.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 157.0);
    }

    #[test]
    fn test_pool_stress_case_158() {
        let input = Tensor::full(vec![1, 1, 4, 4], 158.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 158.0);
    }

    #[test]
    fn test_pool_stress_case_159() {
        let input = Tensor::full(vec![1, 1, 4, 4], 159.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 159.0);
    }

    #[test]
    fn test_pool_stress_case_160() {
        let input = Tensor::full(vec![1, 1, 4, 4], 160.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 160.0);
    }

    #[test]
    fn test_pool_stress_case_161() {
        let input = Tensor::full(vec![1, 1, 4, 4], 161.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 161.0);
    }

    #[test]
    fn test_pool_stress_case_162() {
        let input = Tensor::full(vec![1, 1, 4, 4], 162.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 162.0);
    }

    #[test]
    fn test_pool_stress_case_163() {
        let input = Tensor::full(vec![1, 1, 4, 4], 163.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 163.0);
    }

    #[test]
    fn test_pool_stress_case_164() {
        let input = Tensor::full(vec![1, 1, 4, 4], 164.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 164.0);
    }

    #[test]
    fn test_pool_stress_case_165() {
        let input = Tensor::full(vec![1, 1, 4, 4], 165.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 165.0);
    }

    #[test]
    fn test_pool_stress_case_166() {
        let input = Tensor::full(vec![1, 1, 4, 4], 166.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 166.0);
    }

    #[test]
    fn test_pool_stress_case_167() {
        let input = Tensor::full(vec![1, 1, 4, 4], 167.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 167.0);
    }

    #[test]
    fn test_pool_stress_case_168() {
        let input = Tensor::full(vec![1, 1, 4, 4], 168.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 168.0);
    }

    #[test]
    fn test_pool_stress_case_169() {
        let input = Tensor::full(vec![1, 1, 4, 4], 169.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 169.0);
    }

    #[test]
    fn test_pool_stress_case_170() {
        let input = Tensor::full(vec![1, 1, 4, 4], 170.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 170.0);
    }

    #[test]
    fn test_pool_stress_case_171() {
        let input = Tensor::full(vec![1, 1, 4, 4], 171.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 171.0);
    }

    #[test]
    fn test_pool_stress_case_172() {
        let input = Tensor::full(vec![1, 1, 4, 4], 172.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 172.0);
    }

    #[test]
    fn test_pool_stress_case_173() {
        let input = Tensor::full(vec![1, 1, 4, 4], 173.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 173.0);
    }

    #[test]
    fn test_pool_stress_case_174() {
        let input = Tensor::full(vec![1, 1, 4, 4], 174.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 174.0);
    }

    #[test]
    fn test_pool_stress_case_175() {
        let input = Tensor::full(vec![1, 1, 4, 4], 175.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 175.0);
    }

    #[test]
    fn test_pool_stress_case_176() {
        let input = Tensor::full(vec![1, 1, 4, 4], 176.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 176.0);
    }

    #[test]
    fn test_pool_stress_case_177() {
        let input = Tensor::full(vec![1, 1, 4, 4], 177.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 177.0);
    }

    #[test]
    fn test_pool_stress_case_178() {
        let input = Tensor::full(vec![1, 1, 4, 4], 178.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 178.0);
    }

    #[test]
    fn test_pool_stress_case_179() {
        let input = Tensor::full(vec![1, 1, 4, 4], 179.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 179.0);
    }

    #[test]
    fn test_pool_stress_case_180() {
        let input = Tensor::full(vec![1, 1, 4, 4], 180.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 180.0);
    }

    #[test]
    fn test_pool_stress_case_181() {
        let input = Tensor::full(vec![1, 1, 4, 4], 181.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 181.0);
    }

    #[test]
    fn test_pool_stress_case_182() {
        let input = Tensor::full(vec![1, 1, 4, 4], 182.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 182.0);
    }

    #[test]
    fn test_pool_stress_case_183() {
        let input = Tensor::full(vec![1, 1, 4, 4], 183.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 183.0);
    }

    #[test]
    fn test_pool_stress_case_184() {
        let input = Tensor::full(vec![1, 1, 4, 4], 184.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 184.0);
    }

    #[test]
    fn test_pool_stress_case_185() {
        let input = Tensor::full(vec![1, 1, 4, 4], 185.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 185.0);
    }

    #[test]
    fn test_pool_stress_case_186() {
        let input = Tensor::full(vec![1, 1, 4, 4], 186.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 186.0);
    }

    #[test]
    fn test_pool_stress_case_187() {
        let input = Tensor::full(vec![1, 1, 4, 4], 187.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 187.0);
    }

    #[test]
    fn test_pool_stress_case_188() {
        let input = Tensor::full(vec![1, 1, 4, 4], 188.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 188.0);
    }

    #[test]
    fn test_pool_stress_case_189() {
        let input = Tensor::full(vec![1, 1, 4, 4], 189.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 189.0);
    }

    #[test]
    fn test_pool_stress_case_190() {
        let input = Tensor::full(vec![1, 1, 4, 4], 190.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 190.0);
    }

    #[test]
    fn test_pool_stress_case_191() {
        let input = Tensor::full(vec![1, 1, 4, 4], 191.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 191.0);
    }

    #[test]
    fn test_pool_stress_case_192() {
        let input = Tensor::full(vec![1, 1, 4, 4], 192.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 192.0);
    }

    #[test]
    fn test_pool_stress_case_193() {
        let input = Tensor::full(vec![1, 1, 4, 4], 193.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 193.0);
    }

    #[test]
    fn test_pool_stress_case_194() {
        let input = Tensor::full(vec![1, 1, 4, 4], 194.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 194.0);
    }

    #[test]
    fn test_pool_stress_case_195() {
        let input = Tensor::full(vec![1, 1, 4, 4], 195.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 195.0);
    }

    #[test]
    fn test_pool_stress_case_196() {
        let input = Tensor::full(vec![1, 1, 4, 4], 196.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 196.0);
    }

    #[test]
    fn test_pool_stress_case_197() {
        let input = Tensor::full(vec![1, 1, 4, 4], 197.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 197.0);
    }

    #[test]
    fn test_pool_stress_case_198() {
        let input = Tensor::full(vec![1, 1, 4, 4], 198.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 198.0);
    }

    #[test]
    fn test_pool_stress_case_199() {
        let input = Tensor::full(vec![1, 1, 4, 4], 199.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 199.0);
    }

    #[test]
    fn test_pool_stress_case_200() {
        let input = Tensor::full(vec![1, 1, 4, 4], 200.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 200.0);
    }

    #[test]
    fn test_pool_stress_case_201() {
        let input = Tensor::full(vec![1, 1, 4, 4], 201.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 201.0);
    }

    #[test]
    fn test_pool_stress_case_202() {
        let input = Tensor::full(vec![1, 1, 4, 4], 202.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 202.0);
    }

    #[test]
    fn test_pool_stress_case_203() {
        let input = Tensor::full(vec![1, 1, 4, 4], 203.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 203.0);
    }

    #[test]
    fn test_pool_stress_case_204() {
        let input = Tensor::full(vec![1, 1, 4, 4], 204.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 204.0);
    }

    #[test]
    fn test_pool_stress_case_205() {
        let input = Tensor::full(vec![1, 1, 4, 4], 205.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 205.0);
    }

    #[test]
    fn test_pool_stress_case_206() {
        let input = Tensor::full(vec![1, 1, 4, 4], 206.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 206.0);
    }

    #[test]
    fn test_pool_stress_case_207() {
        let input = Tensor::full(vec![1, 1, 4, 4], 207.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 207.0);
    }

    #[test]
    fn test_pool_stress_case_208() {
        let input = Tensor::full(vec![1, 1, 4, 4], 208.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 208.0);
    }

    #[test]
    fn test_pool_stress_case_209() {
        let input = Tensor::full(vec![1, 1, 4, 4], 209.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 209.0);
    }

    #[test]
    fn test_pool_stress_case_210() {
        let input = Tensor::full(vec![1, 1, 4, 4], 210.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 210.0);
    }

    #[test]
    fn test_pool_stress_case_211() {
        let input = Tensor::full(vec![1, 1, 4, 4], 211.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 211.0);
    }

    #[test]
    fn test_pool_stress_case_212() {
        let input = Tensor::full(vec![1, 1, 4, 4], 212.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 212.0);
    }

    #[test]
    fn test_pool_stress_case_213() {
        let input = Tensor::full(vec![1, 1, 4, 4], 213.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 213.0);
    }

    #[test]
    fn test_pool_stress_case_214() {
        let input = Tensor::full(vec![1, 1, 4, 4], 214.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 214.0);
    }

    #[test]
    fn test_pool_stress_case_215() {
        let input = Tensor::full(vec![1, 1, 4, 4], 215.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 215.0);
    }

    #[test]
    fn test_pool_stress_case_216() {
        let input = Tensor::full(vec![1, 1, 4, 4], 216.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 216.0);
    }

    #[test]
    fn test_pool_stress_case_217() {
        let input = Tensor::full(vec![1, 1, 4, 4], 217.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 217.0);
    }

    #[test]
    fn test_pool_stress_case_218() {
        let input = Tensor::full(vec![1, 1, 4, 4], 218.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 218.0);
    }

    #[test]
    fn test_pool_stress_case_219() {
        let input = Tensor::full(vec![1, 1, 4, 4], 219.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 219.0);
    }

    #[test]
    fn test_pool_stress_case_220() {
        let input = Tensor::full(vec![1, 1, 4, 4], 220.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 220.0);
    }

    #[test]
    fn test_pool_stress_case_221() {
        let input = Tensor::full(vec![1, 1, 4, 4], 221.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 221.0);
    }

    #[test]
    fn test_pool_stress_case_222() {
        let input = Tensor::full(vec![1, 1, 4, 4], 222.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 222.0);
    }

    #[test]
    fn test_pool_stress_case_223() {
        let input = Tensor::full(vec![1, 1, 4, 4], 223.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 223.0);
    }

    #[test]
    fn test_pool_stress_case_224() {
        let input = Tensor::full(vec![1, 1, 4, 4], 224.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 224.0);
    }

    #[test]
    fn test_pool_stress_case_225() {
        let input = Tensor::full(vec![1, 1, 4, 4], 225.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 225.0);
    }

    #[test]
    fn test_pool_stress_case_226() {
        let input = Tensor::full(vec![1, 1, 4, 4], 226.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 226.0);
    }

    #[test]
    fn test_pool_stress_case_227() {
        let input = Tensor::full(vec![1, 1, 4, 4], 227.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 227.0);
    }

    #[test]
    fn test_pool_stress_case_228() {
        let input = Tensor::full(vec![1, 1, 4, 4], 228.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 228.0);
    }

    #[test]
    fn test_pool_stress_case_229() {
        let input = Tensor::full(vec![1, 1, 4, 4], 229.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 229.0);
    }

    #[test]
    fn test_pool_stress_case_230() {
        let input = Tensor::full(vec![1, 1, 4, 4], 230.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 230.0);
    }

    #[test]
    fn test_pool_stress_case_231() {
        let input = Tensor::full(vec![1, 1, 4, 4], 231.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 231.0);
    }

    #[test]
    fn test_pool_stress_case_232() {
        let input = Tensor::full(vec![1, 1, 4, 4], 232.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 232.0);
    }

    #[test]
    fn test_pool_stress_case_233() {
        let input = Tensor::full(vec![1, 1, 4, 4], 233.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 233.0);
    }

    #[test]
    fn test_pool_stress_case_234() {
        let input = Tensor::full(vec![1, 1, 4, 4], 234.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 234.0);
    }

    #[test]
    fn test_pool_stress_case_235() {
        let input = Tensor::full(vec![1, 1, 4, 4], 235.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 235.0);
    }

    #[test]
    fn test_pool_stress_case_236() {
        let input = Tensor::full(vec![1, 1, 4, 4], 236.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 236.0);
    }

    #[test]
    fn test_pool_stress_case_237() {
        let input = Tensor::full(vec![1, 1, 4, 4], 237.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 237.0);
    }

    #[test]
    fn test_pool_stress_case_238() {
        let input = Tensor::full(vec![1, 1, 4, 4], 238.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 238.0);
    }

    #[test]
    fn test_pool_stress_case_239() {
        let input = Tensor::full(vec![1, 1, 4, 4], 239.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 239.0);
    }

    #[test]
    fn test_pool_stress_case_240() {
        let input = Tensor::full(vec![1, 1, 4, 4], 240.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 240.0);
    }

    #[test]
    fn test_pool_stress_case_241() {
        let input = Tensor::full(vec![1, 1, 4, 4], 241.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 241.0);
    }

    #[test]
    fn test_pool_stress_case_242() {
        let input = Tensor::full(vec![1, 1, 4, 4], 242.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 242.0);
    }

    #[test]
    fn test_pool_stress_case_243() {
        let input = Tensor::full(vec![1, 1, 4, 4], 243.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 243.0);
    }

    #[test]
    fn test_pool_stress_case_244() {
        let input = Tensor::full(vec![1, 1, 4, 4], 244.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 244.0);
    }

    #[test]
    fn test_pool_stress_case_245() {
        let input = Tensor::full(vec![1, 1, 4, 4], 245.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 245.0);
    }

    #[test]
    fn test_pool_stress_case_246() {
        let input = Tensor::full(vec![1, 1, 4, 4], 246.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 246.0);
    }

    #[test]
    fn test_pool_stress_case_247() {
        let input = Tensor::full(vec![1, 1, 4, 4], 247.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 247.0);
    }

    #[test]
    fn test_pool_stress_case_248() {
        let input = Tensor::full(vec![1, 1, 4, 4], 248.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 248.0);
    }

    #[test]
    fn test_pool_stress_case_249() {
        let input = Tensor::full(vec![1, 1, 4, 4], 249.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 249.0);
    }

    #[test]
    fn test_pool_stress_case_250() {
        let input = Tensor::full(vec![1, 1, 4, 4], 250.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 250.0);
    }

    #[test]
    fn test_pool_stress_case_251() {
        let input = Tensor::full(vec![1, 1, 4, 4], 251.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 251.0);
    }

    #[test]
    fn test_pool_stress_case_252() {
        let input = Tensor::full(vec![1, 1, 4, 4], 252.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 252.0);
    }

    #[test]
    fn test_pool_stress_case_253() {
        let input = Tensor::full(vec![1, 1, 4, 4], 253.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 253.0);
    }

    #[test]
    fn test_pool_stress_case_254() {
        let input = Tensor::full(vec![1, 1, 4, 4], 254.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 254.0);
    }

    #[test]
    fn test_pool_stress_case_255() {
        let input = Tensor::full(vec![1, 1, 4, 4], 255.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 255.0);
    }

    #[test]
    fn test_pool_stress_case_256() {
        let input = Tensor::full(vec![1, 1, 4, 4], 256.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 256.0);
    }

    #[test]
    fn test_pool_stress_case_257() {
        let input = Tensor::full(vec![1, 1, 4, 4], 257.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 257.0);
    }

    #[test]
    fn test_pool_stress_case_258() {
        let input = Tensor::full(vec![1, 1, 4, 4], 258.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 258.0);
    }

    #[test]
    fn test_pool_stress_case_259() {
        let input = Tensor::full(vec![1, 1, 4, 4], 259.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 259.0);
    }

    #[test]
    fn test_pool_stress_case_260() {
        let input = Tensor::full(vec![1, 1, 4, 4], 260.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 260.0);
    }

    #[test]
    fn test_pool_stress_case_261() {
        let input = Tensor::full(vec![1, 1, 4, 4], 261.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 261.0);
    }

    #[test]
    fn test_pool_stress_case_262() {
        let input = Tensor::full(vec![1, 1, 4, 4], 262.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 262.0);
    }

    #[test]
    fn test_pool_stress_case_263() {
        let input = Tensor::full(vec![1, 1, 4, 4], 263.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 263.0);
    }

    #[test]
    fn test_pool_stress_case_264() {
        let input = Tensor::full(vec![1, 1, 4, 4], 264.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 264.0);
    }

    #[test]
    fn test_pool_stress_case_265() {
        let input = Tensor::full(vec![1, 1, 4, 4], 265.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 265.0);
    }

    #[test]
    fn test_pool_stress_case_266() {
        let input = Tensor::full(vec![1, 1, 4, 4], 266.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 266.0);
    }

    #[test]
    fn test_pool_stress_case_267() {
        let input = Tensor::full(vec![1, 1, 4, 4], 267.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 267.0);
    }

    #[test]
    fn test_pool_stress_case_268() {
        let input = Tensor::full(vec![1, 1, 4, 4], 268.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 268.0);
    }

    #[test]
    fn test_pool_stress_case_269() {
        let input = Tensor::full(vec![1, 1, 4, 4], 269.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 269.0);
    }

    #[test]
    fn test_pool_stress_case_270() {
        let input = Tensor::full(vec![1, 1, 4, 4], 270.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 270.0);
    }

    #[test]
    fn test_pool_stress_case_271() {
        let input = Tensor::full(vec![1, 1, 4, 4], 271.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 271.0);
    }

    #[test]
    fn test_pool_stress_case_272() {
        let input = Tensor::full(vec![1, 1, 4, 4], 272.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 272.0);
    }

    #[test]
    fn test_pool_stress_case_273() {
        let input = Tensor::full(vec![1, 1, 4, 4], 273.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 273.0);
    }

    #[test]
    fn test_pool_stress_case_274() {
        let input = Tensor::full(vec![1, 1, 4, 4], 274.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 274.0);
    }

    #[test]
    fn test_pool_stress_case_275() {
        let input = Tensor::full(vec![1, 1, 4, 4], 275.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 275.0);
    }

    #[test]
    fn test_pool_stress_case_276() {
        let input = Tensor::full(vec![1, 1, 4, 4], 276.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 276.0);
    }

    #[test]
    fn test_pool_stress_case_277() {
        let input = Tensor::full(vec![1, 1, 4, 4], 277.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 277.0);
    }

    #[test]
    fn test_pool_stress_case_278() {
        let input = Tensor::full(vec![1, 1, 4, 4], 278.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 278.0);
    }

    #[test]
    fn test_pool_stress_case_279() {
        let input = Tensor::full(vec![1, 1, 4, 4], 279.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 279.0);
    }

    #[test]
    fn test_pool_stress_case_280() {
        let input = Tensor::full(vec![1, 1, 4, 4], 280.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 280.0);
    }

    #[test]
    fn test_pool_stress_case_281() {
        let input = Tensor::full(vec![1, 1, 4, 4], 281.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 281.0);
    }

    #[test]
    fn test_pool_stress_case_282() {
        let input = Tensor::full(vec![1, 1, 4, 4], 282.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 282.0);
    }

    #[test]
    fn test_pool_stress_case_283() {
        let input = Tensor::full(vec![1, 1, 4, 4], 283.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 283.0);
    }

    #[test]
    fn test_pool_stress_case_284() {
        let input = Tensor::full(vec![1, 1, 4, 4], 284.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 284.0);
    }

    #[test]
    fn test_pool_stress_case_285() {
        let input = Tensor::full(vec![1, 1, 4, 4], 285.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 285.0);
    }

    #[test]
    fn test_pool_stress_case_286() {
        let input = Tensor::full(vec![1, 1, 4, 4], 286.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 286.0);
    }

    #[test]
    fn test_pool_stress_case_287() {
        let input = Tensor::full(vec![1, 1, 4, 4], 287.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 287.0);
    }

    #[test]
    fn test_pool_stress_case_288() {
        let input = Tensor::full(vec![1, 1, 4, 4], 288.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 288.0);
    }

    #[test]
    fn test_pool_stress_case_289() {
        let input = Tensor::full(vec![1, 1, 4, 4], 289.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 289.0);
    }

    #[test]
    fn test_pool_stress_case_290() {
        let input = Tensor::full(vec![1, 1, 4, 4], 290.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 290.0);
    }

    #[test]
    fn test_pool_stress_case_291() {
        let input = Tensor::full(vec![1, 1, 4, 4], 291.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 291.0);
    }

    #[test]
    fn test_pool_stress_case_292() {
        let input = Tensor::full(vec![1, 1, 4, 4], 292.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 292.0);
    }

    #[test]
    fn test_pool_stress_case_293() {
        let input = Tensor::full(vec![1, 1, 4, 4], 293.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 293.0);
    }

    #[test]
    fn test_pool_stress_case_294() {
        let input = Tensor::full(vec![1, 1, 4, 4], 294.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 294.0);
    }

    #[test]
    fn test_pool_stress_case_295() {
        let input = Tensor::full(vec![1, 1, 4, 4], 295.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 295.0);
    }

    #[test]
    fn test_pool_stress_case_296() {
        let input = Tensor::full(vec![1, 1, 4, 4], 296.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 296.0);
    }

    #[test]
    fn test_pool_stress_case_297() {
        let input = Tensor::full(vec![1, 1, 4, 4], 297.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 297.0);
    }

    #[test]
    fn test_pool_stress_case_298() {
        let input = Tensor::full(vec![1, 1, 4, 4], 298.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 298.0);
    }

    #[test]
    fn test_pool_stress_case_299() {
        let input = Tensor::full(vec![1, 1, 4, 4], 299.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 299.0);
    }

    #[test]
    fn test_pool_stress_case_300() {
        let input = Tensor::full(vec![1, 1, 4, 4], 300.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 300.0);
    }

    #[test]
    fn test_pool_stress_case_301() {
        let input = Tensor::full(vec![1, 1, 4, 4], 301.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 301.0);
    }

    #[test]
    fn test_pool_stress_case_302() {
        let input = Tensor::full(vec![1, 1, 4, 4], 302.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 302.0);
    }

    #[test]
    fn test_pool_stress_case_303() {
        let input = Tensor::full(vec![1, 1, 4, 4], 303.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 303.0);
    }

    #[test]
    fn test_pool_stress_case_304() {
        let input = Tensor::full(vec![1, 1, 4, 4], 304.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 304.0);
    }

    #[test]
    fn test_pool_stress_case_305() {
        let input = Tensor::full(vec![1, 1, 4, 4], 305.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 305.0);
    }

    #[test]
    fn test_pool_stress_case_306() {
        let input = Tensor::full(vec![1, 1, 4, 4], 306.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 306.0);
    }

    #[test]
    fn test_pool_stress_case_307() {
        let input = Tensor::full(vec![1, 1, 4, 4], 307.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 307.0);
    }

    #[test]
    fn test_pool_stress_case_308() {
        let input = Tensor::full(vec![1, 1, 4, 4], 308.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 308.0);
    }

    #[test]
    fn test_pool_stress_case_309() {
        let input = Tensor::full(vec![1, 1, 4, 4], 309.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 309.0);
    }

    #[test]
    fn test_pool_stress_case_310() {
        let input = Tensor::full(vec![1, 1, 4, 4], 310.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 310.0);
    }

    #[test]
    fn test_pool_stress_case_311() {
        let input = Tensor::full(vec![1, 1, 4, 4], 311.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 311.0);
    }

    #[test]
    fn test_pool_stress_case_312() {
        let input = Tensor::full(vec![1, 1, 4, 4], 312.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 312.0);
    }

    #[test]
    fn test_pool_stress_case_313() {
        let input = Tensor::full(vec![1, 1, 4, 4], 313.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 313.0);
    }

    #[test]
    fn test_pool_stress_case_314() {
        let input = Tensor::full(vec![1, 1, 4, 4], 314.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 314.0);
    }

    #[test]
    fn test_pool_stress_case_315() {
        let input = Tensor::full(vec![1, 1, 4, 4], 315.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 315.0);
    }

    #[test]
    fn test_pool_stress_case_316() {
        let input = Tensor::full(vec![1, 1, 4, 4], 316.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 316.0);
    }

    #[test]
    fn test_pool_stress_case_317() {
        let input = Tensor::full(vec![1, 1, 4, 4], 317.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 317.0);
    }

    #[test]
    fn test_pool_stress_case_318() {
        let input = Tensor::full(vec![1, 1, 4, 4], 318.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 318.0);
    }

    #[test]
    fn test_pool_stress_case_319() {
        let input = Tensor::full(vec![1, 1, 4, 4], 319.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 319.0);
    }

    #[test]
    fn test_pool_stress_case_320() {
        let input = Tensor::full(vec![1, 1, 4, 4], 320.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 320.0);
    }

    #[test]
    fn test_pool_stress_case_321() {
        let input = Tensor::full(vec![1, 1, 4, 4], 321.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 321.0);
    }

    #[test]
    fn test_pool_stress_case_322() {
        let input = Tensor::full(vec![1, 1, 4, 4], 322.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 322.0);
    }

    #[test]
    fn test_pool_stress_case_323() {
        let input = Tensor::full(vec![1, 1, 4, 4], 323.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 323.0);
    }

    #[test]
    fn test_pool_stress_case_324() {
        let input = Tensor::full(vec![1, 1, 4, 4], 324.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 324.0);
    }

    #[test]
    fn test_pool_stress_case_325() {
        let input = Tensor::full(vec![1, 1, 4, 4], 325.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 325.0);
    }

    #[test]
    fn test_pool_stress_case_326() {
        let input = Tensor::full(vec![1, 1, 4, 4], 326.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 326.0);
    }

    #[test]
    fn test_pool_stress_case_327() {
        let input = Tensor::full(vec![1, 1, 4, 4], 327.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 327.0);
    }

    #[test]
    fn test_pool_stress_case_328() {
        let input = Tensor::full(vec![1, 1, 4, 4], 328.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 328.0);
    }

    #[test]
    fn test_pool_stress_case_329() {
        let input = Tensor::full(vec![1, 1, 4, 4], 329.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 329.0);
    }

    #[test]
    fn test_pool_stress_case_330() {
        let input = Tensor::full(vec![1, 1, 4, 4], 330.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 330.0);
    }

    #[test]
    fn test_pool_stress_case_331() {
        let input = Tensor::full(vec![1, 1, 4, 4], 331.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 331.0);
    }

    #[test]
    fn test_pool_stress_case_332() {
        let input = Tensor::full(vec![1, 1, 4, 4], 332.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 332.0);
    }

    #[test]
    fn test_pool_stress_case_333() {
        let input = Tensor::full(vec![1, 1, 4, 4], 333.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 333.0);
    }

    #[test]
    fn test_pool_stress_case_334() {
        let input = Tensor::full(vec![1, 1, 4, 4], 334.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 334.0);
    }

    #[test]
    fn test_pool_stress_case_335() {
        let input = Tensor::full(vec![1, 1, 4, 4], 335.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 335.0);
    }

    #[test]
    fn test_pool_stress_case_336() {
        let input = Tensor::full(vec![1, 1, 4, 4], 336.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 336.0);
    }

    #[test]
    fn test_pool_stress_case_337() {
        let input = Tensor::full(vec![1, 1, 4, 4], 337.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 337.0);
    }

    #[test]
    fn test_pool_stress_case_338() {
        let input = Tensor::full(vec![1, 1, 4, 4], 338.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 338.0);
    }

    #[test]
    fn test_pool_stress_case_339() {
        let input = Tensor::full(vec![1, 1, 4, 4], 339.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 339.0);
    }

    #[test]
    fn test_pool_stress_case_340() {
        let input = Tensor::full(vec![1, 1, 4, 4], 340.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 340.0);
    }

    #[test]
    fn test_pool_stress_case_341() {
        let input = Tensor::full(vec![1, 1, 4, 4], 341.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 341.0);
    }

    #[test]
    fn test_pool_stress_case_342() {
        let input = Tensor::full(vec![1, 1, 4, 4], 342.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 342.0);
    }

    #[test]
    fn test_pool_stress_case_343() {
        let input = Tensor::full(vec![1, 1, 4, 4], 343.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 343.0);
    }

    #[test]
    fn test_pool_stress_case_344() {
        let input = Tensor::full(vec![1, 1, 4, 4], 344.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 344.0);
    }

    #[test]
    fn test_pool_stress_case_345() {
        let input = Tensor::full(vec![1, 1, 4, 4], 345.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 345.0);
    }

    #[test]
    fn test_pool_stress_case_346() {
        let input = Tensor::full(vec![1, 1, 4, 4], 346.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 346.0);
    }

    #[test]
    fn test_pool_stress_case_347() {
        let input = Tensor::full(vec![1, 1, 4, 4], 347.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 347.0);
    }

    #[test]
    fn test_pool_stress_case_348() {
        let input = Tensor::full(vec![1, 1, 4, 4], 348.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 348.0);
    }

    #[test]
    fn test_pool_stress_case_349() {
        let input = Tensor::full(vec![1, 1, 4, 4], 349.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 349.0);
    }

    #[test]
    fn test_pool_stress_case_350() {
        let input = Tensor::full(vec![1, 1, 4, 4], 350.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 350.0);
    }

    #[test]
    fn test_pool_stress_case_351() {
        let input = Tensor::full(vec![1, 1, 4, 4], 351.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 351.0);
    }

    #[test]
    fn test_pool_stress_case_352() {
        let input = Tensor::full(vec![1, 1, 4, 4], 352.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 352.0);
    }

    #[test]
    fn test_pool_stress_case_353() {
        let input = Tensor::full(vec![1, 1, 4, 4], 353.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 353.0);
    }

    #[test]
    fn test_pool_stress_case_354() {
        let input = Tensor::full(vec![1, 1, 4, 4], 354.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 354.0);
    }

    #[test]
    fn test_pool_stress_case_355() {
        let input = Tensor::full(vec![1, 1, 4, 4], 355.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 355.0);
    }

    #[test]
    fn test_pool_stress_case_356() {
        let input = Tensor::full(vec![1, 1, 4, 4], 356.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 356.0);
    }

    #[test]
    fn test_pool_stress_case_357() {
        let input = Tensor::full(vec![1, 1, 4, 4], 357.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 357.0);
    }

    #[test]
    fn test_pool_stress_case_358() {
        let input = Tensor::full(vec![1, 1, 4, 4], 358.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 358.0);
    }

    #[test]
    fn test_pool_stress_case_359() {
        let input = Tensor::full(vec![1, 1, 4, 4], 359.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 359.0);
    }

    #[test]
    fn test_pool_stress_case_360() {
        let input = Tensor::full(vec![1, 1, 4, 4], 360.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 360.0);
    }

    #[test]
    fn test_pool_stress_case_361() {
        let input = Tensor::full(vec![1, 1, 4, 4], 361.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 361.0);
    }

    #[test]
    fn test_pool_stress_case_362() {
        let input = Tensor::full(vec![1, 1, 4, 4], 362.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 362.0);
    }

    #[test]
    fn test_pool_stress_case_363() {
        let input = Tensor::full(vec![1, 1, 4, 4], 363.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 363.0);
    }

    #[test]
    fn test_pool_stress_case_364() {
        let input = Tensor::full(vec![1, 1, 4, 4], 364.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 364.0);
    }

    #[test]
    fn test_pool_stress_case_365() {
        let input = Tensor::full(vec![1, 1, 4, 4], 365.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 365.0);
    }

    #[test]
    fn test_pool_stress_case_366() {
        let input = Tensor::full(vec![1, 1, 4, 4], 366.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 366.0);
    }

    #[test]
    fn test_pool_stress_case_367() {
        let input = Tensor::full(vec![1, 1, 4, 4], 367.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 367.0);
    }

    #[test]
    fn test_pool_stress_case_368() {
        let input = Tensor::full(vec![1, 1, 4, 4], 368.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 368.0);
    }

    #[test]
    fn test_pool_stress_case_369() {
        let input = Tensor::full(vec![1, 1, 4, 4], 369.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 369.0);
    }

    #[test]
    fn test_pool_stress_case_370() {
        let input = Tensor::full(vec![1, 1, 4, 4], 370.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 370.0);
    }

    #[test]
    fn test_pool_stress_case_371() {
        let input = Tensor::full(vec![1, 1, 4, 4], 371.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 371.0);
    }

    #[test]
    fn test_pool_stress_case_372() {
        let input = Tensor::full(vec![1, 1, 4, 4], 372.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 372.0);
    }

    #[test]
    fn test_pool_stress_case_373() {
        let input = Tensor::full(vec![1, 1, 4, 4], 373.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 373.0);
    }

    #[test]
    fn test_pool_stress_case_374() {
        let input = Tensor::full(vec![1, 1, 4, 4], 374.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 374.0);
    }

    #[test]
    fn test_pool_stress_case_375() {
        let input = Tensor::full(vec![1, 1, 4, 4], 375.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 375.0);
    }

    #[test]
    fn test_pool_stress_case_376() {
        let input = Tensor::full(vec![1, 1, 4, 4], 376.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 376.0);
    }

    #[test]
    fn test_pool_stress_case_377() {
        let input = Tensor::full(vec![1, 1, 4, 4], 377.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 377.0);
    }

    #[test]
    fn test_pool_stress_case_378() {
        let input = Tensor::full(vec![1, 1, 4, 4], 378.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 378.0);
    }

    #[test]
    fn test_pool_stress_case_379() {
        let input = Tensor::full(vec![1, 1, 4, 4], 379.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 379.0);
    }

    #[test]
    fn test_pool_stress_case_380() {
        let input = Tensor::full(vec![1, 1, 4, 4], 380.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 380.0);
    }

    #[test]
    fn test_pool_stress_case_381() {
        let input = Tensor::full(vec![1, 1, 4, 4], 381.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 381.0);
    }

    #[test]
    fn test_pool_stress_case_382() {
        let input = Tensor::full(vec![1, 1, 4, 4], 382.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 382.0);
    }

    #[test]
    fn test_pool_stress_case_383() {
        let input = Tensor::full(vec![1, 1, 4, 4], 383.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 383.0);
    }

    #[test]
    fn test_pool_stress_case_384() {
        let input = Tensor::full(vec![1, 1, 4, 4], 384.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 384.0);
    }

    #[test]
    fn test_pool_stress_case_385() {
        let input = Tensor::full(vec![1, 1, 4, 4], 385.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 385.0);
    }

    #[test]
    fn test_pool_stress_case_386() {
        let input = Tensor::full(vec![1, 1, 4, 4], 386.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 386.0);
    }

    #[test]
    fn test_pool_stress_case_387() {
        let input = Tensor::full(vec![1, 1, 4, 4], 387.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 387.0);
    }

    #[test]
    fn test_pool_stress_case_388() {
        let input = Tensor::full(vec![1, 1, 4, 4], 388.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 388.0);
    }

    #[test]
    fn test_pool_stress_case_389() {
        let input = Tensor::full(vec![1, 1, 4, 4], 389.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 389.0);
    }

    #[test]
    fn test_pool_stress_case_390() {
        let input = Tensor::full(vec![1, 1, 4, 4], 390.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 390.0);
    }

    #[test]
    fn test_pool_stress_case_391() {
        let input = Tensor::full(vec![1, 1, 4, 4], 391.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 391.0);
    }

    #[test]
    fn test_pool_stress_case_392() {
        let input = Tensor::full(vec![1, 1, 4, 4], 392.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 392.0);
    }

    #[test]
    fn test_pool_stress_case_393() {
        let input = Tensor::full(vec![1, 1, 4, 4], 393.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 393.0);
    }

    #[test]
    fn test_pool_stress_case_394() {
        let input = Tensor::full(vec![1, 1, 4, 4], 394.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 394.0);
    }

    #[test]
    fn test_pool_stress_case_395() {
        let input = Tensor::full(vec![1, 1, 4, 4], 395.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 395.0);
    }

    #[test]
    fn test_pool_stress_case_396() {
        let input = Tensor::full(vec![1, 1, 4, 4], 396.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 396.0);
    }

    #[test]
    fn test_pool_stress_case_397() {
        let input = Tensor::full(vec![1, 1, 4, 4], 397.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 397.0);
    }

    #[test]
    fn test_pool_stress_case_398() {
        let input = Tensor::full(vec![1, 1, 4, 4], 398.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 398.0);
    }

    #[test]
    fn test_pool_stress_case_399() {
        let input = Tensor::full(vec![1, 1, 4, 4], 399.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 399.0);
    }

    #[test]
    fn test_pool_stress_case_400() {
        let input = Tensor::full(vec![1, 1, 4, 4], 400.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 400.0);
    }

    #[test]
    fn test_pool_stress_case_401() {
        let input = Tensor::full(vec![1, 1, 4, 4], 401.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 401.0);
    }

    #[test]
    fn test_pool_stress_case_402() {
        let input = Tensor::full(vec![1, 1, 4, 4], 402.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 402.0);
    }

    #[test]
    fn test_pool_stress_case_403() {
        let input = Tensor::full(vec![1, 1, 4, 4], 403.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 403.0);
    }

    #[test]
    fn test_pool_stress_case_404() {
        let input = Tensor::full(vec![1, 1, 4, 4], 404.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 404.0);
    }

    #[test]
    fn test_pool_stress_case_405() {
        let input = Tensor::full(vec![1, 1, 4, 4], 405.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 405.0);
    }

    #[test]
    fn test_pool_stress_case_406() {
        let input = Tensor::full(vec![1, 1, 4, 4], 406.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 406.0);
    }

    #[test]
    fn test_pool_stress_case_407() {
        let input = Tensor::full(vec![1, 1, 4, 4], 407.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 407.0);
    }

    #[test]
    fn test_pool_stress_case_408() {
        let input = Tensor::full(vec![1, 1, 4, 4], 408.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 408.0);
    }

    #[test]
    fn test_pool_stress_case_409() {
        let input = Tensor::full(vec![1, 1, 4, 4], 409.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 409.0);
    }

    #[test]
    fn test_pool_stress_case_410() {
        let input = Tensor::full(vec![1, 1, 4, 4], 410.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 410.0);
    }

    #[test]
    fn test_pool_stress_case_411() {
        let input = Tensor::full(vec![1, 1, 4, 4], 411.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 411.0);
    }

    #[test]
    fn test_pool_stress_case_412() {
        let input = Tensor::full(vec![1, 1, 4, 4], 412.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 412.0);
    }

    #[test]
    fn test_pool_stress_case_413() {
        let input = Tensor::full(vec![1, 1, 4, 4], 413.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 413.0);
    }

    #[test]
    fn test_pool_stress_case_414() {
        let input = Tensor::full(vec![1, 1, 4, 4], 414.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 414.0);
    }

    #[test]
    fn test_pool_stress_case_415() {
        let input = Tensor::full(vec![1, 1, 4, 4], 415.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 415.0);
    }

    #[test]
    fn test_pool_stress_case_416() {
        let input = Tensor::full(vec![1, 1, 4, 4], 416.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 416.0);
    }

    #[test]
    fn test_pool_stress_case_417() {
        let input = Tensor::full(vec![1, 1, 4, 4], 417.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 417.0);
    }

    #[test]
    fn test_pool_stress_case_418() {
        let input = Tensor::full(vec![1, 1, 4, 4], 418.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 418.0);
    }

    #[test]
    fn test_pool_stress_case_419() {
        let input = Tensor::full(vec![1, 1, 4, 4], 419.0);
        let out = global_avg_pool2d(&input);
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert_eq!(out.get(0), 419.0);
    }
}
