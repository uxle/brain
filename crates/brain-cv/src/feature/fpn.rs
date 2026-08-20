//! # Feature Pyramid Network (FPN)
//!
//! Top-down pathway and lateral 1x1 convolutions constructing multi-scale P2–P5 representations (Lin et al. 2017).

use brain_core::Tensor;

/// Feature Pyramid Network (FPN).
#[derive(Clone)]
pub struct Fpn {
    pub in_channels_list: Vec<usize>,
    pub out_channels: usize,
    pub lateral_convs: Vec<Tensor>,
    pub output_convs: Vec<Tensor>,
}

impl Fpn {
    /// Creates a new `Fpn` module.
    pub fn new(in_channels_list: Vec<usize>, out_channels: usize) -> Self {
        let mut lateral_convs = Vec::with_capacity(in_channels_list.len());
        let mut output_convs = Vec::with_capacity(in_channels_list.len());

        for &in_c in &in_channels_list {
            // Lateral 1x1 conv weight: [out_channels, in_c, 1, 1]
            lateral_convs.push(Tensor::ones(vec![out_channels, in_c, 1, 1]));
            // Output 3x3 smooth conv weight: [out_channels, out_channels, 3, 3]
            output_convs.push(Tensor::ones(vec![out_channels, out_channels, 3, 3]));
        }

        Self {
            in_channels_list,
            out_channels,
            lateral_convs,
            output_convs,
        }
    }

    /// Forward pass generating multi-scale feature pyramids.
    /// - `features`: list of bottom-up feature maps [C2, C3, C4, C5] with decreasing spatial resolutions.
    pub fn forward(&self, features: &[Tensor]) -> Vec<Tensor> {
        let num_stages = features.len();
        if num_stages == 0 {
            return Vec::new();
        }

        // Step 1: Compute lateral 1x1 projections
        let mut laterals = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let lat = brain_core::tensor::conv::conv2d(
                &features[i],
                &self.lateral_convs[i],
                None,
                (1, 1),
                (0, 0),
            );
            laterals.push(lat);
        }

        // Step 2: Top-down merging via upsampling
        let mut top_down = laterals.clone();
        for i in (0..num_stages - 1).rev() {
            let current_shape = top_down[i].shape();
            let (target_h, target_w) = (current_shape[2], current_shape[3]);

            let upsampled = upsample_2d_nearest(&top_down[i + 1], target_h, target_w);
            top_down[i] = &top_down[i] + &upsampled;
        }

        // Step 3: 3x3 smooth convolutions
        let mut pyramids = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let p = brain_core::tensor::conv::conv2d_ext(
                &top_down[i],
                &self.output_convs[i],
                None,
                (1, 1),
                (1, 1),
                (1, 1),
            );
            pyramids.push(p);
        }

        pyramids
    }
}

fn upsample_2d_nearest(input: &Tensor, target_h: usize, target_w: usize) -> Tensor {
    let shape = input.shape();
    let (n, c, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

    let mut out_data = vec![0.0f64; n * c * target_h * target_w];

    for b in 0..n {
        for ch in 0..c {
            for y in 0..target_h {
                let src_y = (y * in_h) / target_h;
                for x in 0..target_w {
                    let src_x = (x * in_w) / target_w;
                    let val = input.get_4d(b, ch, src_y, src_x);
                    let out_idx = b * (c * target_h * target_w)
                        + ch * (target_h * target_w)
                        + y * target_w
                        + x;
                    out_data[out_idx] = val;
                }
            }
        }
    }

    Tensor::from_slice(&out_data, vec![n, c, target_h, target_w])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fpn_forward() {
        let fpn = Fpn::new(vec![64, 128], 32);
        let c1 = Tensor::zeros(vec![1, 64, 16, 16]);
        let c2 = Tensor::zeros(vec![1, 128, 8, 8]);
        let res = fpn.forward(&[c1, c2]);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].shape(), &[1, 32, 16, 16]);
        assert_eq!(res[1].shape(), &[1, 32, 8, 8]);
    }
}
