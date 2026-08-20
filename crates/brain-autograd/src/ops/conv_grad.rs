//! # Convolutional Operation Gradients
//!
//! Backward rules for 1D, 2D, and 3D convolutions.

use crate::grad_fns::GradFn;
use crate::value::Value;
use brain_core::{BrainResult, Tensor};
use std::sync::Arc;

/// Differentiable 2D convolution forward operation with stride, padding, and dilation.
pub fn conv2d_ext(
    input: &Value,
    weight: &Value,
    bias: Option<&Value>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Value {
    let bias_tensor = bias.map(|b| b.data());
    let out_tensor = brain_core::tensor::conv::conv2d_ext(
        input.data(),
        weight.data(),
        bias_tensor,
        stride,
        padding,
        dilation,
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
            dilation,
        }
    } else {
        GradFn::None
    };

    Value::from_op(out_tensor, grad_fn, requires_grad)
}

/// Differentiable 2D convolution forward operation (default dilation = (1, 1)).
pub fn conv2d(
    input: &Value,
    weight: &Value,
    bias: Option<&Value>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Value {
    conv2d_ext(input, weight, bias, stride, padding, (1, 1))
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
    dilation: (usize, usize),
) -> BrainResult<(Tensor, Tensor, Option<Tensor>)> {
    assert_eq!(input.ndim(), 4, "conv2d input must be 4D (N, C, H, W)");
    assert_eq!(weight.ndim(), 4, "conv2d weight must be 4D (O, C, KH, KW)");
    assert_eq!(
        grad_output.ndim(),
        4,
        "conv2d grad_output must be 4D (N, O, H_out, W_out)"
    );

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
    let (dh, dw) = (dilation.0.max(1), dilation.1.max(1));

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
                            let ih = h_start + (fh * dh) as isize;
                            if ih >= 0 && (ih as usize) < in_h {
                                let ih = ih as usize;
                                for fw in 0..kw {
                                    let iw = w_start + (fw * dw) as isize;
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
    assert_eq!(
        grad_output.ndim(),
        4,
        "conv_transpose2d grad_output must be 4D (N, C_out, H_out, W_out)"
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
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
