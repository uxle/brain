//! # Depthwise-Separable Convolutions
//!
//! MobileNet-style depthwise spatial convolution followed by 1x1 pointwise projection (Howard et al. 2017).

use brain_core::Tensor;

/// Depthwise-Separable 2D Convolution.
#[derive(Clone)]
pub struct DepthwiseSeparableConv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub depthwise_weight: Tensor,
    pub pointwise_weight: Tensor,
}

impl DepthwiseSeparableConv2d {
    /// Creates a new `DepthwiseSeparableConv2d` layer.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            depthwise_weight: Tensor::ones(vec![in_channels, 1, kernel_size, kernel_size]),
            pointwise_weight: Tensor::ones(vec![out_channels, in_channels, 1, 1]),
        }
    }

    /// Forward pass through depthwise and pointwise convolution stages.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert_eq!(
            shape.len(),
            4,
            "DepthwiseSeparableConv2d expects 4D input [N, C, H, W]"
        );
        let (n, in_c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let pad = self.kernel_size / 2;

        let mut dw_channels = Vec::with_capacity(in_c);
        for c in 0..in_c {
            let mut ch_data = vec![0.0f64; n * h * w];
            for b in 0..n {
                for y in 0..h {
                    for x in 0..w {
                        ch_data[b * (h * w) + y * w + x] = input.get_4d(b, c, y, x);
                    }
                }
            }
            let ch_tensor = Tensor::from_slice(&ch_data, vec![n, 1, h, w]);

            let mut k_data = vec![0.0f64; self.kernel_size * self.kernel_size];
            for ky in 0..self.kernel_size {
                for kx in 0..self.kernel_size {
                    k_data[ky * self.kernel_size + kx] = self.depthwise_weight.get_4d(c, 0, ky, kx);
                }
            }
            let k_tensor =
                Tensor::from_slice(&k_data, vec![1, 1, self.kernel_size, self.kernel_size]);

            let ch_out = brain_core::tensor::conv::conv2d_ext(
                &ch_tensor,
                &k_tensor,
                None,
                (1, 1),
                (pad, pad),
                (1, 1),
            );
            dw_channels.push(ch_out);
        }

        let dw_refs: Vec<&Tensor> = dw_channels.iter().collect();
        let dw_out = brain_core::tensor::ops_nd::cat(&dw_refs, 1);

        brain_core::tensor::conv::conv2d(&dw_out, &self.pointwise_weight, None, (1, 1), (0, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depthwise_separable_forward() {
        let dw = DepthwiseSeparableConv2d::new(3, 8, 3);
        let x = Tensor::zeros(vec![1, 3, 16, 16]);
        let y = dw.forward(&x);
        assert_eq!(y.shape(), &[1, 8, 16, 16]);
    }
}
