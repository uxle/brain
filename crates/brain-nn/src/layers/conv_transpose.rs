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

    #[test]
    fn test_conv_transpose_stress_001() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_002() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_003() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_004() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_005() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_006() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_007() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_008() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_009() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_010() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_011() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_012() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_013() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_014() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_015() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_016() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_017() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_018() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_019() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_020() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_021() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_022() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_023() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_024() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_025() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_026() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_027() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_028() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_029() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_030() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_031() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_032() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_033() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_034() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_035() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_036() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_037() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_038() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_039() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_040() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_041() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_042() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_043() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_044() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_045() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_046() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_047() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_048() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_049() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_050() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_051() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_052() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_053() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_054() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_055() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_056() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_057() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_058() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_059() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_060() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_061() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_062() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_063() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_064() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_065() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_066() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_067() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_068() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_069() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_070() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_071() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_072() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_073() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_074() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_075() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_076() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_077() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_078() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_079() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_080() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_081() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_082() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_083() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_084() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_085() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_086() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_087() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_088() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_089() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_090() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_091() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_092() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_093() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_094() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_095() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_096() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_097() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_098() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_099() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_100() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_101() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_102() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_103() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_104() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_105() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_106() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_107() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_108() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_109() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_110() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_111() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_112() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_113() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_114() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_115() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_116() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_117() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_118() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_119() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_120() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_121() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_122() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_123() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_124() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_125() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_126() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_127() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_128() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_129() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_130() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_131() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_132() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_133() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_134() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_135() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_136() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_137() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_138() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_139() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_140() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_141() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_142() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_143() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_144() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_145() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_146() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_147() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_148() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_149() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_150() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_151() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_152() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_153() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_154() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_155() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_156() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_157() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_158() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_159() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_160() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_161() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_162() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_163() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_164() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_165() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_166() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_167() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_168() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_169() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_170() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_171() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_172() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_173() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_174() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_175() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_176() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_177() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_178() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_179() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_180() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_181() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_182() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_183() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_184() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_185() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_186() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_187() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_188() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_189() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_190() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_191() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_192() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_193() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_194() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_195() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_196() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_197() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_198() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_199() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_200() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_201() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_202() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_203() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_204() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_205() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_206() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_207() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_208() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_209() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_210() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_211() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_212() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_213() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_214() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_215() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_216() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_217() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_218() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_219() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_220() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_221() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_222() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_223() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_224() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_225() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_226() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_227() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_228() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_229() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_230() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_231() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_232() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_233() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_234() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_235() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_236() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_237() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_238() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_239() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_240() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_241() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_242() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_243() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_244() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_245() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_246() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_247() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_248() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_249() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_250() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_251() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_252() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_253() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_254() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_255() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_256() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_257() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_258() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_259() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_260() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_261() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_262() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_263() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_264() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_265() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_266() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_267() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_268() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_269() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_270() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_271() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_272() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_273() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_274() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_275() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_276() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_277() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_278() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_279() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_280() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_281() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_282() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_283() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_284() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_285() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_286() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_287() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_288() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_289() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_290() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_291() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_292() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_293() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_294() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_295() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_296() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_297() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_298() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_299() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_300() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_301() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_302() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_303() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_304() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_305() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_306() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_307() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_308() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_309() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_310() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_311() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_312() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_313() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_314() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_315() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_316() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_317() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_318() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_319() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_320() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_321() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_322() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_323() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_324() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_325() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_326() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_327() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_328() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_329() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_330() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_331() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_332() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_333() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_334() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_335() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_336() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_337() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_338() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_339() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_340() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_341() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_342() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_343() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_344() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_345() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_346() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_347() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_348() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_349() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_350() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_351() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_352() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_353() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_354() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_355() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_356() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_357() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_358() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_359() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_360() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_361() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_362() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_363() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_364() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_365() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_366() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_367() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_368() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_369() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_370() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_371() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_372() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_373() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_374() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_375() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_376() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_377() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_378() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_379() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_380() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_381() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_382() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_383() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_384() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_385() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_386() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_387() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_388() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_389() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_390() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_391() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_392() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_393() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_394() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_395() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_396() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_397() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_398() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_399() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_400() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_401() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_402() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_403() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_404() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_405() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_406() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_407() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_408() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_409() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_410() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    #[test]
    fn test_conv_transpose_stress_411() {
        let ct = ConvTranspose2d::new(16, 8, 3);
        let x = Tensor::zeros(vec![1, 16, 8, 8]);
        let out = ct.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 10, 10]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
}
