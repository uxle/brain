//! # Transposed 2D Convolution (Deconvolution)
//!
//! Fractionally-strided spatial upsampling layer with output padding.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult, ModuleError};

/// Configuration for transposed 2D convolutions.
#[derive(Debug, Clone, Default)]
pub struct ConvTransposeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

/// Transposed 2D Convolution layer.
#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub config: ConvTransposeConfig,
}

impl ConvTranspose2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let weight = Tensor::zeros(vec![in_channels, out_channels, kernel_size, kernel_size]);
        let config = ConvTransposeConfig {
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride: (1, 1),
            padding: (0, 0),
        };
        Self { weight, bias: None, config }
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 || shape[1] != self.config.in_channels {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![shape[0], self.config.in_channels, shape[2], shape[3]],
                got: shape.to_vec(),
            });
        }

        let batch = shape[0];
        let in_h = shape[2];
        let in_w = shape[3];
        let in_c = self.config.in_channels;
        let out_c = self.config.out_channels;
        let (kh, kw) = self.config.kernel_size;
        let (sh, sw) = self.config.stride;
        let (ph, pw) = self.config.padding;

        // Transposed-conv output size: (in - 1)*stride - 2*pad + (kernel - 1) + 1
        let out_h = (in_h - 1) * sh - 2 * ph + kh;
        let out_w = (in_w - 1) * sw - 2 * pw + kw;
        if out_h == 0 || out_w == 0 {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![batch, out_c, out_h, out_w],
                got: shape.to_vec(),
            });
        }

        let mat = out_h * out_w;
        let n_stride = out_c * mat;
        let c_stride = mat;
        let out_numel = batch * n_stride;
        let mut out_data = vec![0.0f64; out_numel];

        // Broadcast bias into every spatial position of its output channel.
        if let Some(ref bias) = self.bias {
            let b_data = bias.to_vec();
            debug_assert_eq!(b_data.len(), out_c, "ConvTranspose2d bias length mismatch");
            for nb in 0..batch {
                for oc in 0..out_c {
                    let base = nb * n_stride + oc * c_stride;
                    let val = b_data[oc];
                    for oh in 0..out_h {
                        let row_off = base + oh * out_w;
                        for ow in 0..out_w {
                            out_data[row_off + ow] += val;
                        }
                    }
                }
            }
        }

        let w_data = self.weight.to_vec(); // [in_c, out_c, kh, kw]
        let w_mat = kh * kw;
        let w_c_stride = out_c * w_mat;
        let in_data = input.to_vec();
        let in_mat = in_h * in_w;
        let in_c_stride = in_mat;

        // Input element (ic, iih, iww) contributes to output position
        // (h_out, w_out) = (iih*sh + fh - ph, iww*sw + fw - pw).
        for nb in 0..batch {
            for ic in 0..in_c {
                let in_base = nb * in_c_stride + ic * in_mat;
                for iih in 0..in_h {
                    for iww in 0..in_w {
                        let in_val = in_data[in_base + iih * in_w + iww];
                        for fh in 0..kh {
                            let h_out = (iih * sh + fh) as isize - ph as isize;
                            if h_out < 0 || h_out as usize >= out_h {
                                continue;
                            }
                            let h_out = h_out as usize;
                            for fw in 0..kw {
                                let w_out = (iww * sw + fw) as isize - pw as isize;
                                if w_out < 0 || w_out as usize >= out_w {
                                    continue;
                                }
                                let w_out = w_out as usize;
                                let out_off = nb * n_stride + h_out * out_w + w_out;
                                let w_base = ic * w_c_stride + fh * kw + fw;
                                for oc in 0..out_c {
                                    out_data[out_off + oc * c_stride] += in_val * w_data[w_base + oc * w_mat];
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(out_data, vec![batch, out_c, out_h, out_w]))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_conv_transpose_correctness() {
        // 1x1 -> 1x1 in=1 out=1, kernel 2, stride 1, pad 0, no bias, weight all ones.
        // input [[1,2],[3,4]]: output is the full overlap sum (3x3):
        // [[1,3,2],[4,10,6],[3,7,4]]
        let mut ct = ConvTranspose2d::new(1, 1, 2);
        ct.weight = Tensor::ones(vec![1, 1, 2, 2]);
        ct.bias = None;
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        let expected = &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (out.get(i) - e).abs() < 1e-9,
                "conv_transpose out[{}] = {} expected {}",
                i,
                out.get(i),
                e
            );
        }
    }
}
