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
    assert_eq!(
        input.ndim(),
        4,
        "conv_transpose2d input must be 4D (N, C_in, H, W)"
    );
    assert_eq!(
        weight.ndim(),
        4,
        "conv_transpose2d weight must be 4D (C_in, C_out, KH, KW)"
    );
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

    let out_h = if in_h == 0 {
        0
    } else {
        ((in_h - 1) * sh + kh).saturating_sub(2 * ph)
    };
    let out_w = if in_w == 0 {
        0
    } else {
        ((in_w - 1) * sw + kw).saturating_sub(2 * pw)
    };

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
    fn test_conv2d_ext_dilated_and_strided() {
        let input = Tensor::ones(vec![1, 1, 5, 5]);
        let weight = Tensor::ones(vec![1, 1, 3, 3]);

        let out_dil = conv2d_ext(&input, &weight, None, (1, 1), (0, 0), (2, 2));
        assert_eq!(out_dil.shape(), &[1, 1, 1, 1]);
        assert_eq!(out_dil.to_vec(), vec![9.0]);

        let out_strided = conv2d_ext(&input, &weight, None, (2, 2), (0, 0), (1, 1));
        assert_eq!(out_strided.shape(), &[1, 1, 2, 2]);
    }
}
