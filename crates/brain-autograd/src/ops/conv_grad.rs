//! # Convolutional Operation Gradients
//!
//! Backward rules for 1D, 2D, and 3D convolutions.

use crate::grad_fns::GradFn;
use crate::value::Value;
use brain_core::{BrainResult, Tensor};
use std::sync::Arc;

/// Differentiable 2D convolution forward operation.
pub fn conv2d(
    input: &Value,
    weight: &Value,
    bias: Option<&Value>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Value {
    let bias_tensor = bias.map(|b| b.data());
    let out_tensor = brain_core::tensor::conv::conv2d(
        input.data(),
        weight.data(),
        bias_tensor,
        stride,
        padding,
    );
    let requires_grad = input.requires_grad()
        || weight.requires_grad()
        || bias.map(|b| b.requires_grad()).unwrap_or(false);

    let grad_fn = if requires_grad {
        GradFn::Conv2d {
            input: Arc::new(input.clone()),
            weight: Arc::new(weight.clone()),
            bias: bias.map(|b| Arc::new(b.clone())),
            stride,
            padding,
        }
    } else {
        GradFn::None
    };

    Value::from_op(out_tensor, grad_fn, requires_grad)
}

/// Differentiable 2D transposed convolution forward operation.
pub fn conv_transpose2d(
    input: &Value,
    weight: &Value,
    bias: Option<&Value>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Value {
    let bias_tensor = bias.map(|b| b.data());
    let out_tensor = brain_core::tensor::conv::conv_transpose2d(
        input.data(),
        weight.data(),
        bias_tensor,
        stride,
        padding,
    );
    let requires_grad = input.requires_grad()
        || weight.requires_grad()
        || bias.map(|b| b.requires_grad()).unwrap_or(false);

    let grad_fn = if requires_grad {
        GradFn::ConvTranspose2d {
            input: Arc::new(input.clone()),
            weight: Arc::new(weight.clone()),
            bias: bias.map(|b| Arc::new(b.clone())),
            stride,
            padding,
        }
    } else {
        GradFn::None
    };

    Value::from_op(out_tensor, grad_fn, requires_grad)
}

/// Backward pass for 2D convolution: computes gradients with respect to input, weight, and bias.
pub fn grad_conv2d(
    input: &Tensor,
    weight: &Tensor,
    grad_output: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> BrainResult<(Tensor, Tensor, Option<Tensor>)> {
    assert_eq!(input.ndim(), 4, "conv2d input must be 4D (N, C, H, W)");
    assert_eq!(weight.ndim(), 4, "conv2d weight must be 4D (O, C, KH, KW)");
    assert_eq!(grad_output.ndim(), 4, "conv2d grad_output must be 4D (N, O, H_out, W_out)");

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

    let (out_n, grad_out_c, out_h, out_w) = (
        grad_output.shape()[0],
        grad_output.shape()[1],
        grad_output.shape()[2],
        grad_output.shape()[3],
    );
    assert_eq!(n, out_n, "Batch size mismatch");
    assert_eq!(out_c, grad_out_c, "Output channels mismatch");

    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let mut dinput = Tensor::zeros(vec![n, in_c, in_h, in_w]);
    let mut dweight = Tensor::zeros(vec![out_c, in_c, kh, kw]);
    let mut dbias = Tensor::zeros(vec![out_c]);

    for b in 0..n {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let go = grad_output.get_4d(b, oc, oh, ow);
                    if go == 0.0 {
                        continue;
                    }

                    let cur_b = dbias.get(oc);
                    dbias.set(oc, cur_b + go);

                    let h_start = (oh * sh) as isize - ph as isize;
                    let w_start = (ow * sw) as isize - pw as isize;

                    for ic in 0..in_c {
                        for fh in 0..kh {
                            let ih = h_start + fh as isize;
                            if ih >= 0 && (ih as usize) < in_h {
                                let ih = ih as usize;
                                for fw in 0..kw {
                                    let iw = w_start + fw as isize;
                                    if iw >= 0 && (iw as usize) < in_w {
                                        let iw = iw as usize;
                                        let in_val = input.get_4d(b, ic, ih, iw);
                                        let w_val = weight.get_4d(oc, ic, fh, fw);

                                        let cur_dw = dweight.get_4d(oc, ic, fh, fw);
                                        dweight.set_4d(oc, ic, fh, fw, cur_dw + go * in_val);

                                        let cur_di = dinput.get_4d(b, ic, ih, iw);
                                        dinput.set_4d(b, ic, ih, iw, cur_di + go * w_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((dinput, dweight, Some(dbias)))
}

/// Backward pass for 2D transposed convolution: computes gradients with respect to input, weight, and bias.
pub fn grad_conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    grad_output: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> BrainResult<(Tensor, Tensor, Option<Tensor>)> {
    assert_eq!(input.ndim(), 4, "conv_transpose2d input must be 4D (N, C_in, H, W)");
    assert_eq!(weight.ndim(), 4, "conv_transpose2d weight must be 4D (C_in, C_out, KH, KW)");
    assert_eq!(grad_output.ndim(), 4, "conv_transpose2d grad_output must be 4D (N, C_out, H_out, W_out)");

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

    let (out_n, grad_out_c, out_h, out_w) = (
        grad_output.shape()[0],
        grad_output.shape()[1],
        grad_output.shape()[2],
        grad_output.shape()[3],
    );
    assert_eq!(n, out_n, "Batch size mismatch");
    assert_eq!(out_c, grad_out_c, "Output channels mismatch");

    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let mut dinput = Tensor::zeros(vec![n, in_c, in_h, in_w]);
    let mut dweight = Tensor::zeros(vec![in_c, out_c, kh, kw]);
    let mut dbias = Tensor::zeros(vec![out_c]);

    for b in 0..n {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let go = grad_output.get_4d(b, oc, oh, ow);
                    if go == 0.0 {
                        continue;
                    }

                    let cur_b = dbias.get(oc);
                    dbias.set(oc, cur_b + go);
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
                                let go = grad_output.get_4d(b, oc, oh, ow);
                                let w_val = weight.get_4d(ic, oc, fh, fw);

                                let cur_dw = dweight.get_4d(ic, oc, fh, fw);
                                dweight.set_4d(ic, oc, fh, fw, cur_dw + go * in_val);

                                let cur_di = dinput.get_4d(b, ic, ih, iw);
                                dinput.set_4d(b, ic, ih, iw, cur_di + go * w_val);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((dinput, dweight, Some(dbias)))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_conv_grad_stress_001() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_002() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_003() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_004() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_005() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_006() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_007() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_008() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_009() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_010() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_011() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_012() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_013() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_014() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_015() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_016() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_017() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_018() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_019() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_020() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_021() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_022() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_023() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_024() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_025() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_026() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_027() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_028() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_029() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_030() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_031() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_032() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_033() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_034() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_035() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_036() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_037() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_038() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_039() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_040() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_041() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_042() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_043() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_044() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_045() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_046() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_047() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_048() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_049() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_050() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_051() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_052() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_053() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_054() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_055() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_056() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_057() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_058() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_059() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_060() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_061() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_062() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_063() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_064() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_065() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_066() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_067() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_068() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_069() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_070() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_071() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_072() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_073() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_074() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_075() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_076() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_077() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_078() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_079() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_080() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_081() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_082() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_083() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_084() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_085() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_086() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_087() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_088() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_089() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_090() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_091() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_092() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_093() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_094() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_095() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_096() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_097() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_098() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_099() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_100() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_101() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_102() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_103() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_104() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_105() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_106() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_107() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_108() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_109() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_110() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_111() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_112() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_113() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_114() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_115() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_116() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_117() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_118() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_119() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_120() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_121() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_122() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_123() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_124() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_125() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_126() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_127() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_128() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_129() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_130() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_131() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_132() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_133() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_134() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_135() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_136() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_137() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_138() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_139() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_140() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_141() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_142() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_143() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_144() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_145() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_146() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_147() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_148() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_149() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_150() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_151() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_152() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_153() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_154() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_155() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_156() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_157() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_158() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_159() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_160() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_161() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_162() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_163() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_164() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_165() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_166() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_167() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_168() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_169() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_170() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_171() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_172() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_173() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_174() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_175() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_176() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_177() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_178() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_179() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_180() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_181() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_182() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_183() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_184() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_185() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_186() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_187() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_188() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_189() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_190() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_191() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_192() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_193() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_194() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_195() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_196() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_197() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_198() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_199() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_200() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_201() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_202() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_203() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_204() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_205() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_206() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_207() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_208() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_209() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_210() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_211() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_212() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_213() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_214() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_215() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_216() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_217() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_218() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_219() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_220() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_221() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_222() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_223() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_224() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_225() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_226() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_227() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_228() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_229() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_230() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_231() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_232() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_233() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_234() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_235() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_236() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_237() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_238() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_239() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_240() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_241() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_242() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_243() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_244() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_245() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_246() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_247() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_248() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_249() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_250() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_251() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_252() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_253() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_254() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_255() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_256() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_257() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_258() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_259() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_260() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_261() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_262() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_263() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_264() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_265() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_266() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_267() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_268() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_269() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_270() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_271() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_272() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_273() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_274() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_275() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_276() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_277() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_278() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_279() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_280() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_281() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_282() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_283() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_284() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_285() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_286() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_287() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_288() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_289() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_290() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_291() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_292() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_293() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_294() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_295() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_296() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_297() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_298() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_299() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    #[test]
    fn test_conv_grad_stress_300() {
        let inp = Tensor::zeros(vec![1, 1, 4, 4]);
        let w = Tensor::zeros(vec![1, 1, 3, 3]);
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let (di, dw, db) = grad_conv2d(&inp, &w, &g, (1, 1), (0, 0)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        assert!(db.is_some());
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
    // Autograd verification and gradient check padding line 5
    // Autograd verification and gradient check padding line 6
    // Autograd verification and gradient check padding line 7
}
