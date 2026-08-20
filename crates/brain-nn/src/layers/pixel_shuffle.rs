//! # PixelShuffle (Sub-Pixel Convolution)
//!
//! Rearranges (B, C*r^2, H, W) tensors into (B, C, H*r, W*r), the sub-pixel
//! upsampling operator used in super-resolution generators.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult, ModuleError};

/// PixelShuffle layer.
#[derive(Debug, Clone, Copy)]
pub struct PixelShuffle {
    pub upscale_factor: usize,
}

impl PixelShuffle {
    pub fn new(upscale_factor: usize) -> Self {
        assert!(upscale_factor > 0, "PixelShuffle upscale factor must be positive");
        Self { upscale_factor }
    }
}

use brain_autograd::Value;

impl Module for PixelShuffle {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let r = self.upscale_factor;
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![1, 1, 1, 1],
                got: shape.to_vec(),
            });
        }
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c % (r * r) != 0 {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![b, r * r, h, w],
                got: shape.to_vec(),
            });
        }
        let c_out = c / (r * r);
        let in_data = input.to_vec();

        let mut out = vec![0.0f64; b * c_out * h * r * w * r];
        for bi in 0..b {
            for co in 0..c_out {
                for hi in 0..h {
                    for wi in 0..w {
                        for ri in 0..r {
                            for rj in 0..r {
                                let c_in = co * r * r + ri * r + rj;
                                let in_idx = ((bi * c + c_in) * h + hi) * w + wi;
                                let h_out = hi * r + ri;
                                let w_out = wi * r + rj;
                                let out_idx = ((bi * c_out + co) * (h * r) + h_out) * (w * r) + w_out;
                                out[out_idx] = in_data[in_idx];
                            }
                        }
                    }
                }
            }
        }

        let t_out = Tensor::from_vec(out, vec![b, c_out, h * r, w * r]);
        Ok(Value::new(t_out, input.requires_grad()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_shuffle_small() {
        let ps = PixelShuffle::new(2);
        // 1 channel, r=2 -> 4 input channels
        let t = Value::new(Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 4, 1, 4],
        ), false);
        let out = ps.forward(&t).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 8]);
        // ch0 (r=0,j=0) fills even rows/cols, ch1 (r=0,j=1) fills odd cols, ch2 (r=1,j=0) fills odd rows
        assert_eq!(out.get(0), 1.0);
        assert_eq!(out.get(1), 5.0);
        assert_eq!(out.get(8), 9.0);
        assert_eq!(out.get(9), 13.0);
        assert_eq!(out.get(7), 8.0);
        assert_eq!(out.get(15), 16.0);
    }

    #[test]
    fn test_pixel_shuffle_roundtrip_permutation() {
        let ps = PixelShuffle::new(3);
        let t = Value::new(Tensor::arange(0.0, 81.0, 1.0).reshape(vec![1, 9, 3, 3]), false);
        let out = ps.forward(&t).unwrap();
        assert_eq!(out.shape(), &[1, 1, 9, 9]);
        // (0,0,0,0) -> (0,0,0,0)
        assert_eq!(out.get(0), 0.0);
        // Original channel 4 = co*9 + 1*3 + 1 -> pixel (1,1): value 4*9=36
        assert_eq!(out.get_4d(0, 0, 1, 1), 36.0);
        // Original channel 8 -> pixel (2,2): value 8*9=72
        assert_eq!(out.get_4d(0, 0, 2, 2), 72.0);
    }
}