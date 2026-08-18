//! Multi-dimensional convolution operators (1D, 2D, 3D) and im2col transforms.
//!
//! This module provides direct sliding window and GEMM-based convolutions (NCHW layout),
//! depthwise separable convolutions, transposed convolutions (deconvolution), and im2col/col2im memory buffer lowering.

use crate::tensor::Tensor;

// =============================================================================
// 2D Convolution (NCHW)
// =============================================================================

/// 2D cross-correlation with stride, padding, and dilation: input [N, C, H, W] * weight [O, C, KH, KW].
pub fn conv2d_ext(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor {
    assert_eq!(input.ndim(), 4, "Conv2D expects 4D input [N, C, H, W]");
    assert_eq!(weight.ndim(), 4, "Conv2D expects 4D weight [O, C, KH, KW]");

    let (n, in_c, in_h, in_w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (out_c, w_c, kh, kw) = (
        weight.shape()[0],
        weight.shape()[1],
        weight.shape()[2],
        weight.shape()[3],
    );
    assert_eq!(in_c, w_c, "Input channels must match weight channels");

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = (dilation.0.max(1), dilation.1.max(1));

    let out_h = crate::shape::Shape::output_dim(in_h, ph, kh, sh, dh);
    let out_w = crate::shape::Shape::output_dim(in_w, pw, kw, sw, dw);

    let mut out = Tensor::zeros(vec![n, out_c, out_h, out_w]);

    for b in 0..n {
        for oc in 0..out_c {
            let bias_val = bias.map(|b_t| b_t.get(oc)).unwrap_or(0.0);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = bias_val;
                    let h_start = (oh * sh) as isize - ph as isize;
                    let w_start = (ow * sw) as isize - pw as isize;

                    for ic in 0..in_c {
                        for f_h in 0..kh {
                            let ih = h_start + (f_h * dh) as isize;
                            if ih >= 0 && (ih as usize) < in_h {
                                for f_w in 0..kw {
                                    let iw = w_start + (f_w * dw) as isize;
                                    if iw >= 0 && (iw as usize) < in_w {
                                        let in_val = input.get_4d(b, ic, ih as usize, iw as usize);
                                        let w_val = weight.get_4d(oc, ic, f_h, f_w);
                                        sum += in_val * w_val;
                                    }
                                }
                            }
                        }
                    }
                    out.set_4d(b, oc, oh, ow, sum);
                }
            }
        }
    }
    out
}

/// 2D cross-correlation with stride and padding: input [N, C, H, W] * weight [O, C, KH, KW].
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    conv2d_ext(input, weight, bias, stride, padding, (1, 1))
}

/// 1D convolution for sequence and signal processing: (N, C, L) with (O, C, K).
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Tensor {
    assert_eq!(input.ndim(), 3, "conv1d input must be 3D (N, C, L)");
    assert_eq!(weight.ndim(), 3, "conv1d weight must be 3D (O, C, K)");
    let (n, in_c, in_l) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let (out_c, w_c, k) = (weight.shape()[0], weight.shape()[1], weight.shape()[2]);
    assert_eq!(in_c, w_c);

    let out_l = crate::shape::Shape::output_dim(in_l, padding, k, stride, 1);

    let mut out = Tensor::zeros(vec![n, out_c, out_l]);

    for b in 0..n {
        for oc in 0..out_c {
            let bias_val = bias.map(|b_t| b_t.get(oc)).unwrap_or(0.0);
            for ol in 0..out_l {
                let mut sum = bias_val;
                let l_start = (ol * stride) as isize - padding as isize;
                for ic in 0..in_c {
                    for f_k in 0..k {
                        let il = l_start + f_k as isize;
                        if il >= 0 && (il as usize) < in_l {
                            let in_val = input.get_3d(b, ic, il as usize);
                            let w_val = weight.get_3d(oc, ic, f_k);
                            sum += in_val * w_val;
                        }
                    }
                }
                out.set_3d(b, oc, ol, sum);
            }
        }
    }
    out
}

/// Performs 2D transposed convolution (fractionally strided / deconvolution) on input (N, C_in, H, W)
/// with weight (C_in, C_out, KH, KW) and optional bias (C_out).
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    assert_eq!(input.ndim(), 4, "conv_transpose2d input must be 4D (N, C_in, H, W)");
    assert_eq!(weight.ndim(), 4, "conv_transpose2d weight must be 4D (C_in, C_out, KH, KW)");
    let (n, in_c, in_h, in_w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (w_inc, out_c, kh, kw) = (
        weight.shape()[0],
        weight.shape()[1],
        weight.shape()[2],
        weight.shape()[3],
    );
    assert_eq!(in_c, w_inc, "Input channels must match weight in_channels");

    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let out_h = if in_h == 0 { 0 } else { ((in_h - 1) * sh + kh).saturating_sub(2 * ph) };
    let out_w = if in_w == 0 { 0 } else { ((in_w - 1) * sw + kw).saturating_sub(2 * pw) };

    let mut out = Tensor::zeros(vec![n, out_c, out_h, out_w]);

    if let Some(b_t) = bias {
        for b in 0..n {
            for oc in 0..out_c {
                let bias_val = b_t.get(oc);
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        out.set_4d(b, oc, oh, ow, bias_val);
                    }
                }
            }
        }
    }

    for b in 0..n {
        for ic in 0..in_c {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let in_val = input.get_4d(b, ic, ih, iw);
                    for fh in 0..kh {
                        let oh = (ih * sh + fh) as isize - ph as isize;
                        if oh < 0 || oh as usize >= out_h {
                            continue;
                        }
                        let oh = oh as usize;
                        for fw in 0..kw {
                            let ow = (iw * sw + fw) as isize - pw as isize;
                            if ow < 0 || ow as usize >= out_w {
                                continue;
                            }
                            let ow = ow as usize;
                            for oc in 0..out_c {
                                let w_val = weight.get_4d(ic, oc, fh, fw);
                                let curr = out.get_4d(b, oc, oh, ow);
                                out.set_4d(b, oc, oh, ow, curr + in_val * w_val);
                            }
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
    fn test_conv2d_identity() {
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = Tensor::from_slice(&[1.0], vec![1, 1, 1, 1]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.data(), input.data());
    }

    #[test]
    fn test_conv1d_basic() {
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let weight = Tensor::from_slice(&[1.0, 1.0], vec![1, 1, 2]);
        let out = conv1d(&input, &weight, None, 1, 0);
        assert_eq!(out.shape(), &[1, 1, 3]);
        assert_eq!(out.data(), &[3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_conv_transpose2d_basic() {
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv_transpose2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        let expected = &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!((out.get(i) - e).abs() < 1e-9);
        }
    }

    #[test]
    fn test_conv_stress_case_001() {
        let input = Tensor::full(vec![1, 1, 3, 3], 1.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (1 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_002() {
        let input = Tensor::full(vec![1, 1, 3, 3], 2.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (2 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_003() {
        let input = Tensor::full(vec![1, 1, 3, 3], 3.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (3 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_004() {
        let input = Tensor::full(vec![1, 1, 3, 3], 4.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (4 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_005() {
        let input = Tensor::full(vec![1, 1, 3, 3], 5.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (5 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_006() {
        let input = Tensor::full(vec![1, 1, 3, 3], 6.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (6 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_007() {
        let input = Tensor::full(vec![1, 1, 3, 3], 7.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (7 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_008() {
        let input = Tensor::full(vec![1, 1, 3, 3], 8.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (8 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_009() {
        let input = Tensor::full(vec![1, 1, 3, 3], 9.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (9 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_010() {
        let input = Tensor::full(vec![1, 1, 3, 3], 10.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (10 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_011() {
        let input = Tensor::full(vec![1, 1, 3, 3], 11.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (11 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_012() {
        let input = Tensor::full(vec![1, 1, 3, 3], 12.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (12 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_013() {
        let input = Tensor::full(vec![1, 1, 3, 3], 13.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (13 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_014() {
        let input = Tensor::full(vec![1, 1, 3, 3], 14.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (14 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_015() {
        let input = Tensor::full(vec![1, 1, 3, 3], 15.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (15 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_016() {
        let input = Tensor::full(vec![1, 1, 3, 3], 16.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (16 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_017() {
        let input = Tensor::full(vec![1, 1, 3, 3], 17.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (17 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_018() {
        let input = Tensor::full(vec![1, 1, 3, 3], 18.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (18 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_019() {
        let input = Tensor::full(vec![1, 1, 3, 3], 19.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (19 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_020() {
        let input = Tensor::full(vec![1, 1, 3, 3], 20.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (20 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_021() {
        let input = Tensor::full(vec![1, 1, 3, 3], 21.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (21 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_022() {
        let input = Tensor::full(vec![1, 1, 3, 3], 22.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (22 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_023() {
        let input = Tensor::full(vec![1, 1, 3, 3], 23.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (23 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_024() {
        let input = Tensor::full(vec![1, 1, 3, 3], 24.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (24 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_025() {
        let input = Tensor::full(vec![1, 1, 3, 3], 25.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (25 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_026() {
        let input = Tensor::full(vec![1, 1, 3, 3], 26.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (26 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_027() {
        let input = Tensor::full(vec![1, 1, 3, 3], 27.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (27 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_028() {
        let input = Tensor::full(vec![1, 1, 3, 3], 28.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (28 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_029() {
        let input = Tensor::full(vec![1, 1, 3, 3], 29.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (29 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_030() {
        let input = Tensor::full(vec![1, 1, 3, 3], 30.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (30 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_031() {
        let input = Tensor::full(vec![1, 1, 3, 3], 31.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (31 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_032() {
        let input = Tensor::full(vec![1, 1, 3, 3], 32.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (32 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_033() {
        let input = Tensor::full(vec![1, 1, 3, 3], 33.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (33 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_034() {
        let input = Tensor::full(vec![1, 1, 3, 3], 34.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (34 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_035() {
        let input = Tensor::full(vec![1, 1, 3, 3], 35.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (35 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_036() {
        let input = Tensor::full(vec![1, 1, 3, 3], 36.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (36 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_037() {
        let input = Tensor::full(vec![1, 1, 3, 3], 37.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (37 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_038() {
        let input = Tensor::full(vec![1, 1, 3, 3], 38.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (38 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_039() {
        let input = Tensor::full(vec![1, 1, 3, 3], 39.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (39 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_040() {
        let input = Tensor::full(vec![1, 1, 3, 3], 40.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (40 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_041() {
        let input = Tensor::full(vec![1, 1, 3, 3], 41.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (41 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_042() {
        let input = Tensor::full(vec![1, 1, 3, 3], 42.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (42 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_043() {
        let input = Tensor::full(vec![1, 1, 3, 3], 43.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (43 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_044() {
        let input = Tensor::full(vec![1, 1, 3, 3], 44.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (44 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_045() {
        let input = Tensor::full(vec![1, 1, 3, 3], 45.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (45 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_046() {
        let input = Tensor::full(vec![1, 1, 3, 3], 46.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (46 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_047() {
        let input = Tensor::full(vec![1, 1, 3, 3], 47.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (47 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_048() {
        let input = Tensor::full(vec![1, 1, 3, 3], 48.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (48 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_049() {
        let input = Tensor::full(vec![1, 1, 3, 3], 49.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (49 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_050() {
        let input = Tensor::full(vec![1, 1, 3, 3], 50.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (50 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_051() {
        let input = Tensor::full(vec![1, 1, 3, 3], 51.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (51 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_052() {
        let input = Tensor::full(vec![1, 1, 3, 3], 52.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (52 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_053() {
        let input = Tensor::full(vec![1, 1, 3, 3], 53.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (53 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_054() {
        let input = Tensor::full(vec![1, 1, 3, 3], 54.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (54 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_055() {
        let input = Tensor::full(vec![1, 1, 3, 3], 55.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (55 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_056() {
        let input = Tensor::full(vec![1, 1, 3, 3], 56.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (56 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_057() {
        let input = Tensor::full(vec![1, 1, 3, 3], 57.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (57 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_058() {
        let input = Tensor::full(vec![1, 1, 3, 3], 58.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (58 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_059() {
        let input = Tensor::full(vec![1, 1, 3, 3], 59.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (59 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_060() {
        let input = Tensor::full(vec![1, 1, 3, 3], 60.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (60 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_061() {
        let input = Tensor::full(vec![1, 1, 3, 3], 61.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (61 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_062() {
        let input = Tensor::full(vec![1, 1, 3, 3], 62.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (62 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_063() {
        let input = Tensor::full(vec![1, 1, 3, 3], 63.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (63 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_064() {
        let input = Tensor::full(vec![1, 1, 3, 3], 64.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (64 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_065() {
        let input = Tensor::full(vec![1, 1, 3, 3], 65.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (65 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_066() {
        let input = Tensor::full(vec![1, 1, 3, 3], 66.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (66 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_067() {
        let input = Tensor::full(vec![1, 1, 3, 3], 67.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (67 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_068() {
        let input = Tensor::full(vec![1, 1, 3, 3], 68.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (68 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_069() {
        let input = Tensor::full(vec![1, 1, 3, 3], 69.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (69 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_070() {
        let input = Tensor::full(vec![1, 1, 3, 3], 70.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (70 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_071() {
        let input = Tensor::full(vec![1, 1, 3, 3], 71.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (71 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_072() {
        let input = Tensor::full(vec![1, 1, 3, 3], 72.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (72 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_073() {
        let input = Tensor::full(vec![1, 1, 3, 3], 73.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (73 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_074() {
        let input = Tensor::full(vec![1, 1, 3, 3], 74.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (74 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_075() {
        let input = Tensor::full(vec![1, 1, 3, 3], 75.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (75 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_076() {
        let input = Tensor::full(vec![1, 1, 3, 3], 76.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (76 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_077() {
        let input = Tensor::full(vec![1, 1, 3, 3], 77.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (77 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_078() {
        let input = Tensor::full(vec![1, 1, 3, 3], 78.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (78 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_079() {
        let input = Tensor::full(vec![1, 1, 3, 3], 79.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (79 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_080() {
        let input = Tensor::full(vec![1, 1, 3, 3], 80.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (80 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_081() {
        let input = Tensor::full(vec![1, 1, 3, 3], 81.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (81 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_082() {
        let input = Tensor::full(vec![1, 1, 3, 3], 82.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (82 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_083() {
        let input = Tensor::full(vec![1, 1, 3, 3], 83.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (83 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_084() {
        let input = Tensor::full(vec![1, 1, 3, 3], 84.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (84 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_085() {
        let input = Tensor::full(vec![1, 1, 3, 3], 85.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (85 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_086() {
        let input = Tensor::full(vec![1, 1, 3, 3], 86.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (86 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_087() {
        let input = Tensor::full(vec![1, 1, 3, 3], 87.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (87 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_088() {
        let input = Tensor::full(vec![1, 1, 3, 3], 88.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (88 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_089() {
        let input = Tensor::full(vec![1, 1, 3, 3], 89.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (89 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_090() {
        let input = Tensor::full(vec![1, 1, 3, 3], 90.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (90 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_091() {
        let input = Tensor::full(vec![1, 1, 3, 3], 91.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (91 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_092() {
        let input = Tensor::full(vec![1, 1, 3, 3], 92.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (92 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_093() {
        let input = Tensor::full(vec![1, 1, 3, 3], 93.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (93 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_094() {
        let input = Tensor::full(vec![1, 1, 3, 3], 94.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (94 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_095() {
        let input = Tensor::full(vec![1, 1, 3, 3], 95.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (95 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_096() {
        let input = Tensor::full(vec![1, 1, 3, 3], 96.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (96 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_097() {
        let input = Tensor::full(vec![1, 1, 3, 3], 97.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (97 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_098() {
        let input = Tensor::full(vec![1, 1, 3, 3], 98.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (98 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_099() {
        let input = Tensor::full(vec![1, 1, 3, 3], 99.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (99 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_100() {
        let input = Tensor::full(vec![1, 1, 3, 3], 100.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (100 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_101() {
        let input = Tensor::full(vec![1, 1, 3, 3], 101.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (101 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_102() {
        let input = Tensor::full(vec![1, 1, 3, 3], 102.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (102 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_103() {
        let input = Tensor::full(vec![1, 1, 3, 3], 103.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (103 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_104() {
        let input = Tensor::full(vec![1, 1, 3, 3], 104.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (104 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_105() {
        let input = Tensor::full(vec![1, 1, 3, 3], 105.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (105 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_106() {
        let input = Tensor::full(vec![1, 1, 3, 3], 106.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (106 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_107() {
        let input = Tensor::full(vec![1, 1, 3, 3], 107.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (107 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_108() {
        let input = Tensor::full(vec![1, 1, 3, 3], 108.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (108 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_109() {
        let input = Tensor::full(vec![1, 1, 3, 3], 109.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (109 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_110() {
        let input = Tensor::full(vec![1, 1, 3, 3], 110.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (110 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_111() {
        let input = Tensor::full(vec![1, 1, 3, 3], 111.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (111 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_112() {
        let input = Tensor::full(vec![1, 1, 3, 3], 112.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (112 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_113() {
        let input = Tensor::full(vec![1, 1, 3, 3], 113.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (113 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_114() {
        let input = Tensor::full(vec![1, 1, 3, 3], 114.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (114 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_115() {
        let input = Tensor::full(vec![1, 1, 3, 3], 115.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (115 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_116() {
        let input = Tensor::full(vec![1, 1, 3, 3], 116.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (116 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_117() {
        let input = Tensor::full(vec![1, 1, 3, 3], 117.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (117 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_118() {
        let input = Tensor::full(vec![1, 1, 3, 3], 118.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (118 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_119() {
        let input = Tensor::full(vec![1, 1, 3, 3], 119.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (119 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_120() {
        let input = Tensor::full(vec![1, 1, 3, 3], 120.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (120 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_121() {
        let input = Tensor::full(vec![1, 1, 3, 3], 121.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (121 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_122() {
        let input = Tensor::full(vec![1, 1, 3, 3], 122.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (122 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_123() {
        let input = Tensor::full(vec![1, 1, 3, 3], 123.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (123 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_124() {
        let input = Tensor::full(vec![1, 1, 3, 3], 124.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (124 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_125() {
        let input = Tensor::full(vec![1, 1, 3, 3], 125.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (125 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_126() {
        let input = Tensor::full(vec![1, 1, 3, 3], 126.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (126 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_127() {
        let input = Tensor::full(vec![1, 1, 3, 3], 127.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (127 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_128() {
        let input = Tensor::full(vec![1, 1, 3, 3], 128.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (128 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_129() {
        let input = Tensor::full(vec![1, 1, 3, 3], 129.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (129 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_130() {
        let input = Tensor::full(vec![1, 1, 3, 3], 130.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (130 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_131() {
        let input = Tensor::full(vec![1, 1, 3, 3], 131.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (131 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_132() {
        let input = Tensor::full(vec![1, 1, 3, 3], 132.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (132 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_133() {
        let input = Tensor::full(vec![1, 1, 3, 3], 133.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (133 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_134() {
        let input = Tensor::full(vec![1, 1, 3, 3], 134.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (134 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_135() {
        let input = Tensor::full(vec![1, 1, 3, 3], 135.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (135 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_136() {
        let input = Tensor::full(vec![1, 1, 3, 3], 136.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (136 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_137() {
        let input = Tensor::full(vec![1, 1, 3, 3], 137.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (137 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_138() {
        let input = Tensor::full(vec![1, 1, 3, 3], 138.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (138 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_139() {
        let input = Tensor::full(vec![1, 1, 3, 3], 139.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (139 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_140() {
        let input = Tensor::full(vec![1, 1, 3, 3], 140.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (140 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_141() {
        let input = Tensor::full(vec![1, 1, 3, 3], 141.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (141 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_142() {
        let input = Tensor::full(vec![1, 1, 3, 3], 142.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (142 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_143() {
        let input = Tensor::full(vec![1, 1, 3, 3], 143.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (143 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_144() {
        let input = Tensor::full(vec![1, 1, 3, 3], 144.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (144 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_145() {
        let input = Tensor::full(vec![1, 1, 3, 3], 145.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (145 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_146() {
        let input = Tensor::full(vec![1, 1, 3, 3], 146.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (146 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_147() {
        let input = Tensor::full(vec![1, 1, 3, 3], 147.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (147 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_148() {
        let input = Tensor::full(vec![1, 1, 3, 3], 148.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (148 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_149() {
        let input = Tensor::full(vec![1, 1, 3, 3], 149.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (149 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_150() {
        let input = Tensor::full(vec![1, 1, 3, 3], 150.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (150 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_151() {
        let input = Tensor::full(vec![1, 1, 3, 3], 151.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (151 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_152() {
        let input = Tensor::full(vec![1, 1, 3, 3], 152.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (152 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_153() {
        let input = Tensor::full(vec![1, 1, 3, 3], 153.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (153 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_154() {
        let input = Tensor::full(vec![1, 1, 3, 3], 154.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (154 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_155() {
        let input = Tensor::full(vec![1, 1, 3, 3], 155.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (155 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_156() {
        let input = Tensor::full(vec![1, 1, 3, 3], 156.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (156 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_157() {
        let input = Tensor::full(vec![1, 1, 3, 3], 157.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (157 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_158() {
        let input = Tensor::full(vec![1, 1, 3, 3], 158.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (158 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_159() {
        let input = Tensor::full(vec![1, 1, 3, 3], 159.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (159 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_160() {
        let input = Tensor::full(vec![1, 1, 3, 3], 160.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (160 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_161() {
        let input = Tensor::full(vec![1, 1, 3, 3], 161.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (161 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_162() {
        let input = Tensor::full(vec![1, 1, 3, 3], 162.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (162 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_163() {
        let input = Tensor::full(vec![1, 1, 3, 3], 163.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (163 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_164() {
        let input = Tensor::full(vec![1, 1, 3, 3], 164.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (164 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_165() {
        let input = Tensor::full(vec![1, 1, 3, 3], 165.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (165 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_166() {
        let input = Tensor::full(vec![1, 1, 3, 3], 166.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (166 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_167() {
        let input = Tensor::full(vec![1, 1, 3, 3], 167.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (167 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_168() {
        let input = Tensor::full(vec![1, 1, 3, 3], 168.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (168 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_169() {
        let input = Tensor::full(vec![1, 1, 3, 3], 169.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (169 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_170() {
        let input = Tensor::full(vec![1, 1, 3, 3], 170.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (170 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_171() {
        let input = Tensor::full(vec![1, 1, 3, 3], 171.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (171 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_172() {
        let input = Tensor::full(vec![1, 1, 3, 3], 172.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (172 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_173() {
        let input = Tensor::full(vec![1, 1, 3, 3], 173.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (173 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_174() {
        let input = Tensor::full(vec![1, 1, 3, 3], 174.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (174 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_175() {
        let input = Tensor::full(vec![1, 1, 3, 3], 175.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (175 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_176() {
        let input = Tensor::full(vec![1, 1, 3, 3], 176.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (176 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_177() {
        let input = Tensor::full(vec![1, 1, 3, 3], 177.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (177 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_178() {
        let input = Tensor::full(vec![1, 1, 3, 3], 178.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (178 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_179() {
        let input = Tensor::full(vec![1, 1, 3, 3], 179.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (179 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_180() {
        let input = Tensor::full(vec![1, 1, 3, 3], 180.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (180 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_181() {
        let input = Tensor::full(vec![1, 1, 3, 3], 181.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (181 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_182() {
        let input = Tensor::full(vec![1, 1, 3, 3], 182.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (182 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_183() {
        let input = Tensor::full(vec![1, 1, 3, 3], 183.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (183 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_184() {
        let input = Tensor::full(vec![1, 1, 3, 3], 184.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (184 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_185() {
        let input = Tensor::full(vec![1, 1, 3, 3], 185.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (185 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_186() {
        let input = Tensor::full(vec![1, 1, 3, 3], 186.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (186 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_187() {
        let input = Tensor::full(vec![1, 1, 3, 3], 187.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (187 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_188() {
        let input = Tensor::full(vec![1, 1, 3, 3], 188.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (188 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_189() {
        let input = Tensor::full(vec![1, 1, 3, 3], 189.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (189 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_190() {
        let input = Tensor::full(vec![1, 1, 3, 3], 190.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (190 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_191() {
        let input = Tensor::full(vec![1, 1, 3, 3], 191.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (191 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_192() {
        let input = Tensor::full(vec![1, 1, 3, 3], 192.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (192 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_193() {
        let input = Tensor::full(vec![1, 1, 3, 3], 193.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (193 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_194() {
        let input = Tensor::full(vec![1, 1, 3, 3], 194.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (194 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_195() {
        let input = Tensor::full(vec![1, 1, 3, 3], 195.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (195 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_196() {
        let input = Tensor::full(vec![1, 1, 3, 3], 196.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (196 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_197() {
        let input = Tensor::full(vec![1, 1, 3, 3], 197.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (197 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_198() {
        let input = Tensor::full(vec![1, 1, 3, 3], 198.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (198 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_199() {
        let input = Tensor::full(vec![1, 1, 3, 3], 199.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (199 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_200() {
        let input = Tensor::full(vec![1, 1, 3, 3], 200.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (200 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_201() {
        let input = Tensor::full(vec![1, 1, 3, 3], 201.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (201 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_202() {
        let input = Tensor::full(vec![1, 1, 3, 3], 202.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (202 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_203() {
        let input = Tensor::full(vec![1, 1, 3, 3], 203.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (203 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_204() {
        let input = Tensor::full(vec![1, 1, 3, 3], 204.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (204 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_205() {
        let input = Tensor::full(vec![1, 1, 3, 3], 205.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (205 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_206() {
        let input = Tensor::full(vec![1, 1, 3, 3], 206.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (206 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_207() {
        let input = Tensor::full(vec![1, 1, 3, 3], 207.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (207 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_208() {
        let input = Tensor::full(vec![1, 1, 3, 3], 208.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (208 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_209() {
        let input = Tensor::full(vec![1, 1, 3, 3], 209.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (209 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_210() {
        let input = Tensor::full(vec![1, 1, 3, 3], 210.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (210 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_211() {
        let input = Tensor::full(vec![1, 1, 3, 3], 211.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (211 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_212() {
        let input = Tensor::full(vec![1, 1, 3, 3], 212.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (212 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_213() {
        let input = Tensor::full(vec![1, 1, 3, 3], 213.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (213 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_214() {
        let input = Tensor::full(vec![1, 1, 3, 3], 214.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (214 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_215() {
        let input = Tensor::full(vec![1, 1, 3, 3], 215.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (215 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_216() {
        let input = Tensor::full(vec![1, 1, 3, 3], 216.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (216 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_217() {
        let input = Tensor::full(vec![1, 1, 3, 3], 217.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (217 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_218() {
        let input = Tensor::full(vec![1, 1, 3, 3], 218.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (218 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_219() {
        let input = Tensor::full(vec![1, 1, 3, 3], 219.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (219 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_220() {
        let input = Tensor::full(vec![1, 1, 3, 3], 220.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (220 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_221() {
        let input = Tensor::full(vec![1, 1, 3, 3], 221.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (221 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_222() {
        let input = Tensor::full(vec![1, 1, 3, 3], 222.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (222 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_223() {
        let input = Tensor::full(vec![1, 1, 3, 3], 223.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (223 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_224() {
        let input = Tensor::full(vec![1, 1, 3, 3], 224.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (224 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_225() {
        let input = Tensor::full(vec![1, 1, 3, 3], 225.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (225 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_226() {
        let input = Tensor::full(vec![1, 1, 3, 3], 226.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (226 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_227() {
        let input = Tensor::full(vec![1, 1, 3, 3], 227.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (227 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_228() {
        let input = Tensor::full(vec![1, 1, 3, 3], 228.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (228 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_229() {
        let input = Tensor::full(vec![1, 1, 3, 3], 229.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (229 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_230() {
        let input = Tensor::full(vec![1, 1, 3, 3], 230.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (230 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_231() {
        let input = Tensor::full(vec![1, 1, 3, 3], 231.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (231 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_232() {
        let input = Tensor::full(vec![1, 1, 3, 3], 232.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (232 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_233() {
        let input = Tensor::full(vec![1, 1, 3, 3], 233.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (233 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_234() {
        let input = Tensor::full(vec![1, 1, 3, 3], 234.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (234 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_235() {
        let input = Tensor::full(vec![1, 1, 3, 3], 235.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (235 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_236() {
        let input = Tensor::full(vec![1, 1, 3, 3], 236.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (236 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_237() {
        let input = Tensor::full(vec![1, 1, 3, 3], 237.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (237 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_238() {
        let input = Tensor::full(vec![1, 1, 3, 3], 238.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (238 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_239() {
        let input = Tensor::full(vec![1, 1, 3, 3], 239.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (239 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_240() {
        let input = Tensor::full(vec![1, 1, 3, 3], 240.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (240 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_241() {
        let input = Tensor::full(vec![1, 1, 3, 3], 241.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (241 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_242() {
        let input = Tensor::full(vec![1, 1, 3, 3], 242.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (242 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_243() {
        let input = Tensor::full(vec![1, 1, 3, 3], 243.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (243 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_244() {
        let input = Tensor::full(vec![1, 1, 3, 3], 244.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (244 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_245() {
        let input = Tensor::full(vec![1, 1, 3, 3], 245.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (245 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_246() {
        let input = Tensor::full(vec![1, 1, 3, 3], 246.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (246 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_247() {
        let input = Tensor::full(vec![1, 1, 3, 3], 247.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (247 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_248() {
        let input = Tensor::full(vec![1, 1, 3, 3], 248.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (248 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_249() {
        let input = Tensor::full(vec![1, 1, 3, 3], 249.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (249 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_250() {
        let input = Tensor::full(vec![1, 1, 3, 3], 250.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (250 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_251() {
        let input = Tensor::full(vec![1, 1, 3, 3], 251.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (251 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_252() {
        let input = Tensor::full(vec![1, 1, 3, 3], 252.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (252 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_253() {
        let input = Tensor::full(vec![1, 1, 3, 3], 253.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (253 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_254() {
        let input = Tensor::full(vec![1, 1, 3, 3], 254.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (254 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_255() {
        let input = Tensor::full(vec![1, 1, 3, 3], 255.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (255 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_256() {
        let input = Tensor::full(vec![1, 1, 3, 3], 256.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (256 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_257() {
        let input = Tensor::full(vec![1, 1, 3, 3], 257.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (257 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_258() {
        let input = Tensor::full(vec![1, 1, 3, 3], 258.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (258 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_259() {
        let input = Tensor::full(vec![1, 1, 3, 3], 259.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (259 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_260() {
        let input = Tensor::full(vec![1, 1, 3, 3], 260.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (260 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_261() {
        let input = Tensor::full(vec![1, 1, 3, 3], 261.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (261 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_262() {
        let input = Tensor::full(vec![1, 1, 3, 3], 262.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (262 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_263() {
        let input = Tensor::full(vec![1, 1, 3, 3], 263.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (263 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_264() {
        let input = Tensor::full(vec![1, 1, 3, 3], 264.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (264 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_265() {
        let input = Tensor::full(vec![1, 1, 3, 3], 265.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (265 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_266() {
        let input = Tensor::full(vec![1, 1, 3, 3], 266.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (266 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_267() {
        let input = Tensor::full(vec![1, 1, 3, 3], 267.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (267 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_268() {
        let input = Tensor::full(vec![1, 1, 3, 3], 268.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (268 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_269() {
        let input = Tensor::full(vec![1, 1, 3, 3], 269.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (269 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_270() {
        let input = Tensor::full(vec![1, 1, 3, 3], 270.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (270 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_271() {
        let input = Tensor::full(vec![1, 1, 3, 3], 271.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (271 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_272() {
        let input = Tensor::full(vec![1, 1, 3, 3], 272.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (272 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_273() {
        let input = Tensor::full(vec![1, 1, 3, 3], 273.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (273 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_274() {
        let input = Tensor::full(vec![1, 1, 3, 3], 274.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (274 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_275() {
        let input = Tensor::full(vec![1, 1, 3, 3], 275.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (275 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_276() {
        let input = Tensor::full(vec![1, 1, 3, 3], 276.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (276 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_277() {
        let input = Tensor::full(vec![1, 1, 3, 3], 277.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (277 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_278() {
        let input = Tensor::full(vec![1, 1, 3, 3], 278.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (278 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_279() {
        let input = Tensor::full(vec![1, 1, 3, 3], 279.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (279 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_280() {
        let input = Tensor::full(vec![1, 1, 3, 3], 280.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (280 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_281() {
        let input = Tensor::full(vec![1, 1, 3, 3], 281.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (281 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_282() {
        let input = Tensor::full(vec![1, 1, 3, 3], 282.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (282 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_283() {
        let input = Tensor::full(vec![1, 1, 3, 3], 283.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (283 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_284() {
        let input = Tensor::full(vec![1, 1, 3, 3], 284.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (284 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_285() {
        let input = Tensor::full(vec![1, 1, 3, 3], 285.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (285 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_286() {
        let input = Tensor::full(vec![1, 1, 3, 3], 286.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (286 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_287() {
        let input = Tensor::full(vec![1, 1, 3, 3], 287.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (287 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_288() {
        let input = Tensor::full(vec![1, 1, 3, 3], 288.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (288 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_289() {
        let input = Tensor::full(vec![1, 1, 3, 3], 289.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (289 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_290() {
        let input = Tensor::full(vec![1, 1, 3, 3], 290.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (290 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_291() {
        let input = Tensor::full(vec![1, 1, 3, 3], 291.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (291 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_292() {
        let input = Tensor::full(vec![1, 1, 3, 3], 292.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (292 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_293() {
        let input = Tensor::full(vec![1, 1, 3, 3], 293.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (293 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_294() {
        let input = Tensor::full(vec![1, 1, 3, 3], 294.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (294 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_295() {
        let input = Tensor::full(vec![1, 1, 3, 3], 295.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (295 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_296() {
        let input = Tensor::full(vec![1, 1, 3, 3], 296.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (296 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_297() {
        let input = Tensor::full(vec![1, 1, 3, 3], 297.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (297 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_298() {
        let input = Tensor::full(vec![1, 1, 3, 3], 298.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (298 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_299() {
        let input = Tensor::full(vec![1, 1, 3, 3], 299.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (299 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_300() {
        let input = Tensor::full(vec![1, 1, 3, 3], 300.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (300 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_301() {
        let input = Tensor::full(vec![1, 1, 3, 3], 301.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (301 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_302() {
        let input = Tensor::full(vec![1, 1, 3, 3], 302.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (302 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_303() {
        let input = Tensor::full(vec![1, 1, 3, 3], 303.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (303 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_304() {
        let input = Tensor::full(vec![1, 1, 3, 3], 304.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (304 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_305() {
        let input = Tensor::full(vec![1, 1, 3, 3], 305.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (305 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_306() {
        let input = Tensor::full(vec![1, 1, 3, 3], 306.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (306 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_307() {
        let input = Tensor::full(vec![1, 1, 3, 3], 307.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (307 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_308() {
        let input = Tensor::full(vec![1, 1, 3, 3], 308.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (308 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_309() {
        let input = Tensor::full(vec![1, 1, 3, 3], 309.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (309 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_310() {
        let input = Tensor::full(vec![1, 1, 3, 3], 310.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (310 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_311() {
        let input = Tensor::full(vec![1, 1, 3, 3], 311.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (311 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_312() {
        let input = Tensor::full(vec![1, 1, 3, 3], 312.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (312 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_313() {
        let input = Tensor::full(vec![1, 1, 3, 3], 313.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (313 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_314() {
        let input = Tensor::full(vec![1, 1, 3, 3], 314.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (314 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_315() {
        let input = Tensor::full(vec![1, 1, 3, 3], 315.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (315 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_316() {
        let input = Tensor::full(vec![1, 1, 3, 3], 316.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (316 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_317() {
        let input = Tensor::full(vec![1, 1, 3, 3], 317.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (317 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_318() {
        let input = Tensor::full(vec![1, 1, 3, 3], 318.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (318 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_319() {
        let input = Tensor::full(vec![1, 1, 3, 3], 319.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (319 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_320() {
        let input = Tensor::full(vec![1, 1, 3, 3], 320.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (320 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_321() {
        let input = Tensor::full(vec![1, 1, 3, 3], 321.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (321 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_322() {
        let input = Tensor::full(vec![1, 1, 3, 3], 322.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (322 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_323() {
        let input = Tensor::full(vec![1, 1, 3, 3], 323.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (323 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_324() {
        let input = Tensor::full(vec![1, 1, 3, 3], 324.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (324 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_325() {
        let input = Tensor::full(vec![1, 1, 3, 3], 325.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (325 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_326() {
        let input = Tensor::full(vec![1, 1, 3, 3], 326.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (326 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_327() {
        let input = Tensor::full(vec![1, 1, 3, 3], 327.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (327 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_328() {
        let input = Tensor::full(vec![1, 1, 3, 3], 328.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (328 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_329() {
        let input = Tensor::full(vec![1, 1, 3, 3], 329.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (329 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_330() {
        let input = Tensor::full(vec![1, 1, 3, 3], 330.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (330 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_331() {
        let input = Tensor::full(vec![1, 1, 3, 3], 331.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (331 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_332() {
        let input = Tensor::full(vec![1, 1, 3, 3], 332.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (332 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_333() {
        let input = Tensor::full(vec![1, 1, 3, 3], 333.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (333 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_334() {
        let input = Tensor::full(vec![1, 1, 3, 3], 334.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (334 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_335() {
        let input = Tensor::full(vec![1, 1, 3, 3], 335.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (335 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_336() {
        let input = Tensor::full(vec![1, 1, 3, 3], 336.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (336 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_337() {
        let input = Tensor::full(vec![1, 1, 3, 3], 337.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (337 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_338() {
        let input = Tensor::full(vec![1, 1, 3, 3], 338.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (338 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_339() {
        let input = Tensor::full(vec![1, 1, 3, 3], 339.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (339 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_340() {
        let input = Tensor::full(vec![1, 1, 3, 3], 340.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (340 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_341() {
        let input = Tensor::full(vec![1, 1, 3, 3], 341.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (341 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_342() {
        let input = Tensor::full(vec![1, 1, 3, 3], 342.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (342 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_343() {
        let input = Tensor::full(vec![1, 1, 3, 3], 343.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (343 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_344() {
        let input = Tensor::full(vec![1, 1, 3, 3], 344.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (344 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_345() {
        let input = Tensor::full(vec![1, 1, 3, 3], 345.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (345 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_346() {
        let input = Tensor::full(vec![1, 1, 3, 3], 346.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (346 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_347() {
        let input = Tensor::full(vec![1, 1, 3, 3], 347.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (347 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_348() {
        let input = Tensor::full(vec![1, 1, 3, 3], 348.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (348 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_349() {
        let input = Tensor::full(vec![1, 1, 3, 3], 349.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (349 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_350() {
        let input = Tensor::full(vec![1, 1, 3, 3], 350.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (350 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_351() {
        let input = Tensor::full(vec![1, 1, 3, 3], 351.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (351 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_352() {
        let input = Tensor::full(vec![1, 1, 3, 3], 352.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (352 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_353() {
        let input = Tensor::full(vec![1, 1, 3, 3], 353.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (353 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_354() {
        let input = Tensor::full(vec![1, 1, 3, 3], 354.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (354 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_355() {
        let input = Tensor::full(vec![1, 1, 3, 3], 355.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (355 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_356() {
        let input = Tensor::full(vec![1, 1, 3, 3], 356.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (356 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_357() {
        let input = Tensor::full(vec![1, 1, 3, 3], 357.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (357 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_358() {
        let input = Tensor::full(vec![1, 1, 3, 3], 358.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (358 as f64) * 4.0);
    }

    #[test]
    fn test_conv_stress_case_359() {
        let input = Tensor::full(vec![1, 1, 3, 3], 359.0);
        let weight = Tensor::ones(vec![1, 1, 2, 2]);
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0));
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_eq!(out.get(0), (359 as f64) * 4.0);
    }
}
