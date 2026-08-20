//! # RoIAlign Layer
//!
//! Extracts feature maps using continuous bilinear grid sampling without spatial quantization artifacts (He et al. 2017).

use brain_core::Tensor;

/// RoIAlign Layer.
#[derive(Clone)]
pub struct RoIAlign {
    pub output_size: (usize, usize),
    pub spatial_scale: f64,
    pub sampling_ratio: usize,
    pub aligned: bool,
}

impl RoIAlign {
    /// Creates a new `RoIAlign` layer.
    pub fn new(output_size: (usize, usize), spatial_scale: f64, sampling_ratio: usize) -> Self {
        Self {
            output_size,
            spatial_scale,
            sampling_ratio,
            aligned: true,
        }
    }

    /// Forward pass sampling features with bilinear interpolation.
    /// - `features`: 4D Tensor [N, C, H, W]
    /// - `rois`: 2D Tensor [K, 5] with columns [batch_idx, x1, y1, x2, y2]
    pub fn forward(&self, features: &Tensor, rois: &Tensor) -> Tensor {
        let f_shape = features.shape();
        assert_eq!(
            f_shape.len(),
            4,
            "RoIAlign expects 4D features [N, C, H, W]"
        );
        let (n_batch, c, feat_h, feat_w) = (f_shape[0], f_shape[1], f_shape[2], f_shape[3]);

        let num_rois = rois.shape()[0];
        let (out_h, out_w) = self.output_size;
        let mut output = vec![0.0f64; num_rois * c * out_h * out_w];

        let offset = if self.aligned { 0.5 } else { 0.0 };

        for k in 0..num_rois {
            let b_idx = (rois.get_2d(k, 0) as usize).min(n_batch.saturating_sub(1));
            let x1 = rois.get_2d(k, 1) * self.spatial_scale - offset;
            let y1 = rois.get_2d(k, 2) * self.spatial_scale - offset;
            let x2 = rois.get_2d(k, 3) * self.spatial_scale - offset;
            let y2 = rois.get_2d(k, 4) * self.spatial_scale - offset;

            let roi_w = (x2 - x1).max(1.0);
            let roi_h = (y2 - y1).max(1.0);
            let bin_size_h = roi_h / (out_h as f64);
            let bin_size_w = roi_w / (out_w as f64);

            let roi_bin_grid_h = if self.sampling_ratio > 0 {
                self.sampling_ratio
            } else {
                (bin_size_h.ceil() as usize).max(1)
            };
            let roi_bin_grid_w = if self.sampling_ratio > 0 {
                self.sampling_ratio
            } else {
                (bin_size_w.ceil() as usize).max(1)
            };
            let count = (roi_bin_grid_h * roi_bin_grid_w) as f64;

            for ch in 0..c {
                for ph in 0..out_h {
                    for pw in 0..out_w {
                        let mut sum_val = 0.0;

                        for iy in 0..roi_bin_grid_h {
                            let y = y1
                                + (ph as f64) * bin_size_h
                                + ((iy as f64 + 0.5) * bin_size_h) / (roi_bin_grid_h as f64);

                            for ix in 0..roi_bin_grid_w {
                                let x = x1
                                    + (pw as f64) * bin_size_w
                                    + ((ix as f64 + 0.5) * bin_size_w) / (roi_bin_grid_w as f64);

                                sum_val += bilinear_interpolate_4d(
                                    features, b_idx, ch, feat_h, feat_w, y, x,
                                );
                            }
                        }

                        let out_idx =
                            k * (c * out_h * out_w) + ch * (out_h * out_w) + ph * out_w + pw;
                        output[out_idx] = sum_val / count;
                    }
                }
            }
        }

        Tensor::from_slice(&output, vec![num_rois, c, out_h, out_w])
    }
}

fn bilinear_interpolate_4d(
    features: &Tensor,
    b: usize,
    c: usize,
    height: usize,
    width: usize,
    y: f64,
    x: f64,
) -> f64 {
    if y < -1.0 || y > height as f64 || x < -1.0 || x > width as f64 {
        return 0.0;
    }
    let y = y.clamp(0.0, (height - 1) as f64);
    let x = x.clamp(0.0, (width - 1) as f64);

    let y_low = y.floor() as usize;
    let x_low = x.floor() as usize;
    let y_high = (y_low + 1).min(height - 1);
    let x_high = (x_low + 1).min(width - 1);

    let ly = y - y_low as f64;
    let lx = x - x_low as f64;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    let v1 = features.get_4d(b, c, y_low, x_low);
    let v2 = features.get_4d(b, c, y_low, x_high);
    let v3 = features.get_4d(b, c, y_high, x_low);
    let v4 = features.get_4d(b, c, y_high, x_high);

    (hy * hx) * v1 + (hy * lx) * v2 + (ly * hx) * v3 + (ly * lx) * v4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_align_basic() {
        let roi = RoIAlign::new((7, 7), 1.0, 2);
        let mut features = Tensor::zeros(vec![1, 1, 14, 14]);
        features.set_4d(0, 0, 5, 5, 10.0);

        let rois = Tensor::from_slice(&[0.0, 0.0, 0.0, 14.0, 14.0], vec![1, 5]);
        let aligned = roi.forward(&features, &rois);
        assert_eq!(aligned.shape(), &[1, 1, 7, 7]);
    }
}
