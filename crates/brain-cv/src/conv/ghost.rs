//! # Ghost Convolution Modules
//!
//! GhostModule architecture generating more features with cheap linear transformation operations (Han et al. 2020).

use brain_core::Tensor;

/// Ghost Convolution Module.
#[derive(Clone)]
pub struct GhostModule {
    pub in_channels: usize,
    pub out_channels: usize,
    pub primary_conv_weight: Tensor,
    pub cheap_conv_weight: Tensor,
}

impl GhostModule {
    /// Creates a new `GhostModule`.
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let init_channels = (out_channels + 1) / 2;
        Self {
            in_channels,
            out_channels,
            primary_conv_weight: Tensor::ones(vec![init_channels, in_channels, 1, 1]),
            cheap_conv_weight: Tensor::ones(vec![init_channels, 1, 3, 3]),
        }
    }

    /// Forward pass concatenating primary and intrinsic ghost feature maps.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert_eq!(shape.len(), 4, "GhostModule expects 4D input [N, C, H, W]");
        let (n, _, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        let init_c = (self.out_channels + 1) / 2;
        let ghost_c = self.out_channels - init_c;

        // Primary 1x1 conv features
        let primary = brain_core::tensor::conv::conv2d(
            input,
            &self.primary_conv_weight,
            None,
            (1, 1),
            (0, 0),
        );

        // Cheap depthwise 3x3 operation with padding 1
        let mut cheap_channels = Vec::with_capacity(ghost_c);
        for c in 0..ghost_c {
            let mut ch_data = vec![0.0f64; n * h * w];
            for b in 0..n {
                for y in 0..h {
                    for x in 0..w {
                        ch_data[b * (h * w) + y * w + x] = primary.get_4d(b, c, y, x);
                    }
                }
            }
            let ch_tensor = Tensor::from_slice(&ch_data, vec![n, 1, h, w]);

            let mut k_data = vec![0.0f64; 9];
            for ky in 0..3 {
                for kx in 0..3 {
                    k_data[ky * 3 + kx] = self.cheap_conv_weight.get_4d(c % init_c, 0, ky, kx);
                }
            }
            let k_tensor = Tensor::from_slice(&k_data, vec![1, 1, 3, 3]);

            let ch_out = brain_core::tensor::conv::conv2d_ext(
                &ch_tensor,
                &k_tensor,
                None,
                (1, 1),
                (1, 1),
                (1, 1),
            );
            cheap_channels.push(ch_out);
        }

        let cheap_refs: Vec<&Tensor> = cheap_channels.iter().collect();
        let cheap = brain_core::tensor::ops_nd::cat(&cheap_refs, 1);

        // Concatenate primary and cheap features along channel dimension (dim=1)
        brain_core::tensor::ops_nd::cat(&[&primary, &cheap], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghost_module_forward() {
        let ghost = GhostModule::new(8, 16);
        let x = Tensor::zeros(vec![1, 8, 16, 16]);
        let y = ghost.forward(&x);
        assert_eq!(y.shape(), &[1, 16, 16, 16]);
    }
}
