//! # Pooling Operation Gradients
//!
//! Backward rules for Max Pooling, Average Pooling, and Adaptive Pooling.

use crate::grad_fns::GradFn;
use crate::value::Value;
use brain_core::{BrainResult, Tensor};
use std::sync::Arc;

/// Differentiable 2D Max Pooling forward operation.
pub fn max_pool2d(
    input: &Value,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Value {
    let out_tensor = brain_core::tensor::pool::max_pool2d(
        input.data(),
        kernel_size,
        stride,
        padding,
    );
    let requires_grad = input.requires_grad();
    let grad_fn = if requires_grad {
        GradFn::MaxPool2d {
            input: Arc::new(input.clone()),
            kernel_size,
            stride,
            padding,
        }
    } else {
        GradFn::None
    };

    Value::from_op(out_tensor, grad_fn, requires_grad)
}

/// Differentiable 2D Average Pooling forward operation.
pub fn avg_pool2d(
    input: &Value,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Value {
    let out_tensor = brain_core::tensor::pool::avg_pool2d(
        input.data(),
        kernel_size,
        stride,
        padding,
    );
    let requires_grad = input.requires_grad();
    let grad_fn = if requires_grad {
        GradFn::AvgPool2d {
            input: Arc::new(input.clone()),
            kernel_size,
            stride,
            padding,
        }
    } else {
        GradFn::None
    };

    Value::from_op(out_tensor, grad_fn, requires_grad)
}

/// Backward pass for 2D Average Pooling with stride and padding.
pub fn grad_avg_pool2d_ext(
    input_shape: &[usize],
    grad_output: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> BrainResult<Tensor> {
    assert_eq!(input_shape.len(), 4, "avg_pool2d input must be 4D (N, C, H, W)");
    assert_eq!(grad_output.ndim(), 4, "avg_pool2d grad_output must be 4D (N, C, H_out, W_out)");

    let (n, c, in_h, in_w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (out_n, out_c, out_h, out_w) = (
        grad_output.shape()[0],
        grad_output.shape()[1],
        grad_output.shape()[2],
        grad_output.shape()[3],
    );
    assert_eq!(n, out_n, "Batch size mismatch");
    assert_eq!(c, out_c, "Channel count mismatch");

    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let mut dinput = Tensor::zeros(input_shape.to_vec());
    let scale = 1.0 / (kh * kw) as f64;

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let go = grad_output.get_4d(b, ch, oh, ow);
                    if go == 0.0 {
                        continue;
                    }

                    let h_start = (oh * sh) as isize - ph as isize;
                    let w_start = (ow * sw) as isize - pw as isize;

                    for fh in 0..kh {
                        let ih = h_start + fh as isize;
                        if ih >= 0 && (ih as usize) < in_h {
                            let ih = ih as usize;
                            for fw in 0..kw {
                                let iw = w_start + fw as isize;
                                if iw >= 0 && (iw as usize) < in_w {
                                    let iw = iw as usize;
                                    let cur = dinput.get_4d(b, ch, ih, iw);
                                    dinput.set_4d(b, ch, ih, iw, cur + go * scale);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(dinput)
}

/// Backward pass for 2D Average Pooling (stride = kernel_size, padding = 0).
pub fn grad_avg_pool2d(
    input_shape: &[usize],
    grad_output: &Tensor,
    kernel_size: (usize, usize),
) -> BrainResult<Tensor> {
    grad_avg_pool2d_ext(input_shape, grad_output, kernel_size, kernel_size, (0, 0))
}

/// Backward pass for 2D Max Pooling.
pub fn grad_max_pool2d(
    input: &Tensor,
    grad_output: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> BrainResult<Tensor> {
    assert_eq!(input.ndim(), 4, "max_pool2d input must be 4D (N, C, H, W)");
    assert_eq!(grad_output.ndim(), 4, "max_pool2d grad_output must be 4D (N, C, H_out, W_out)");

    let (n, c, in_h, in_w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (out_n, out_c, out_h, out_w) = (
        grad_output.shape()[0],
        grad_output.shape()[1],
        grad_output.shape()[2],
        grad_output.shape()[3],
    );
    assert_eq!(n, out_n, "Batch size mismatch");
    assert_eq!(c, out_c, "Channel count mismatch");

    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let mut dinput = Tensor::zeros(vec![n, c, in_h, in_w]);

    for b in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let go = grad_output.get_4d(b, ch, oh, ow);
                    if go == 0.0 {
                        continue;
                    }

                    let h_start = (oh * sh) as isize - ph as isize;
                    let w_start = (ow * sw) as isize - pw as isize;

                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_pos = None;

                    for fh in 0..kh {
                        let ih = h_start + fh as isize;
                        if ih >= 0 && (ih as usize) < in_h {
                            let ih = ih as usize;
                            for fw in 0..kw {
                                let iw = w_start + fw as isize;
                                if iw >= 0 && (iw as usize) < in_w {
                                    let iw = iw as usize;
                                    let val = input.get_4d(b, ch, ih, iw);
                                    if val > max_val {
                                        max_val = val;
                                        max_pos = Some((ih, iw));
                                    }
                                }
                            }
                        }
                    }

                    if let Some((max_h, max_w)) = max_pos {
                        let cur = dinput.get_4d(b, ch, max_h, max_w);
                        dinput.set_4d(b, ch, max_h, max_w, cur + go);
                    }
                }
            }
        }
    }

    Ok(dinput)
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
}
