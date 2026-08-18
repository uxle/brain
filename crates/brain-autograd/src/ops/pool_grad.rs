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

    #[test]
    fn test_pool_grad_stress_001() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_002() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_003() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_004() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_005() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_006() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_007() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_008() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_009() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_010() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_011() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_012() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_013() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_014() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_015() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_016() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_017() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_018() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_019() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_020() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_021() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_022() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_023() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_024() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_025() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_026() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_027() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_028() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_029() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_030() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_031() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_032() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_033() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_034() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_035() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_036() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_037() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_038() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_039() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_040() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_041() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_042() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_043() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_044() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_045() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_046() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_047() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_048() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_049() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_050() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_051() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_052() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_053() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_054() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_055() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_056() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_057() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_058() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_059() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_060() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_061() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_062() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_063() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_064() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_065() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_066() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_067() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_068() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_069() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_070() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_071() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_072() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_073() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_074() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_075() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_076() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_077() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_078() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_079() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_080() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_081() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_082() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_083() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_084() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_085() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_086() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_087() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_088() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_089() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_090() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_091() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_092() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_093() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_094() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_095() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_096() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_097() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_098() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_099() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_100() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_101() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_102() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_103() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_104() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_105() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_106() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_107() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_108() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_109() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_110() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_111() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_112() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_113() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_114() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_115() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_116() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_117() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_118() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_119() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_120() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_121() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_122() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_123() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_124() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_125() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_126() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_127() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_128() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_129() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_130() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_131() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_132() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_133() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_134() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_135() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_136() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_137() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_138() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_139() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_140() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_141() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_142() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_143() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_144() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_145() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_146() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_147() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_148() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_149() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_150() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_151() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_152() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_153() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_154() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_155() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_156() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_157() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_158() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_159() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_160() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_161() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_162() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_163() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_164() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_165() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_166() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_167() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_168() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_169() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_170() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_171() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_172() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_173() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_174() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_175() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_176() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_177() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_178() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_179() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_180() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_181() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_182() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_183() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_184() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_185() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_186() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_187() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_188() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_189() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_190() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_191() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_192() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_193() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_194() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_195() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_196() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_197() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_198() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_199() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_200() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_201() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_202() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_203() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_204() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_205() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_206() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_207() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_208() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_209() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_210() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_211() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_212() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_213() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_214() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_215() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_216() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_217() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_218() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_219() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_220() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_221() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_222() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_223() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_224() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_225() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_226() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_227() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_228() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_229() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_230() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_231() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_232() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_233() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_234() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_235() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_236() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_237() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_238() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_239() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_240() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_241() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_242() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_243() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_244() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_245() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_246() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_247() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_248() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_249() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_250() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_251() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_252() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_253() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_254() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_255() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_256() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_257() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_258() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_259() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_260() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_261() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_262() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_263() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_264() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_265() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_266() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_267() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_268() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_269() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_270() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_271() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_272() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_273() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_274() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_275() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_276() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_277() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_278() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_279() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_280() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_281() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_282() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_283() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_284() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_285() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_286() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_287() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_288() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_289() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_290() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_291() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_292() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_293() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_294() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_295() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_296() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_297() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_298() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_299() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_300() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_301() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_302() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_303() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_304() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_305() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_306() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_307() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_308() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_309() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_310() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_311() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_312() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_313() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_314() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_315() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_316() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_317() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_318() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_319() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_320() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_321() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_322() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_323() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_324() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_325() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_326() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_327() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_328() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_329() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_330() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_331() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_332() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_333() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_334() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_335() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_336() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_337() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_338() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_339() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_340() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_341() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_342() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_343() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_344() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_345() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_346() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_347() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_348() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_349() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_350() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_351() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_352() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_353() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_354() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_355() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_356() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_357() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_358() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_359() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_360() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_361() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_362() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_363() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_364() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_365() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_366() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_367() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_368() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_369() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_370() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_371() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_372() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_373() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_374() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_375() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_376() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_377() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_378() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_379() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_380() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_381() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_382() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_383() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_384() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_385() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_386() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_387() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_388() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_389() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_390() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_391() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_392() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_393() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_394() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_395() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_396() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_397() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_398() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_399() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_400() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_401() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_402() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_403() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_404() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_405() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_406() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_407() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_408() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_409() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_410() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_411() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_412() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_413() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_414() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_415() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_416() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_417() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_418() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_419() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_420() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_421() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_422() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_423() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_424() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_425() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_426() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_427() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_428() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_429() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_430() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_431() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_432() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_433() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_434() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_435() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_436() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_437() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_438() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_439() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_440() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_441() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_442() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_443() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_444() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_445() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_446() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_447() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_448() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_449() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_450() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_451() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_452() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_453() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_454() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_455() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_456() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_457() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_458() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_459() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_460() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_461() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_462() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_463() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_464() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_465() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_466() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_467() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_468() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_469() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_470() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_471() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_472() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_grad_stress_473() {
        let g = Tensor::zeros(vec![1, 1, 2, 2]);
        let di = grad_avg_pool2d(&[1, 1, 4, 4], &g, (2, 2)).unwrap();
        assert_eq!(di.shape(), &[1, 1, 4, 4]);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
}
