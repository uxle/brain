//! # Compound Mixing Augmentations (MixUp, CutMix, Mosaic)
//!
//! Multi-image mixing strategies for regularizing vision models during training (Zhang et al. 2017, Yun et al. 2019).

use brain_core::Tensor;

/// Blends two images using linear interpolation (MixUp): out = alpha * img1 + (1 - alpha) * img2.
pub fn mixup(img1: &Tensor, img2: &Tensor, alpha: f64) -> Tensor {
    assert_eq!(
        img1.shape(),
        img2.shape(),
        "img1 and img2 shapes must match"
    );
    let alpha_clamped = alpha.clamp(0.0, 1.0);
    let t_alpha = Tensor::scalar(alpha_clamped);
    let t_inv = Tensor::scalar(1.0 - alpha_clamped);
    &(img1 * &t_alpha) + &(img2 * &t_inv)
}

/// Pastes a rectangular patch from `img2` into `img1` (CutMix).
/// - `bbox`: `[x1, y1, x2, y2]` spatial coordinates of the bounding box to replace.
pub fn cutmix(img1: &Tensor, img2: &Tensor, bbox: &[usize; 4]) -> Tensor {
    assert_eq!(
        img1.shape(),
        img2.shape(),
        "img1 and img2 shapes must match"
    );
    let shape = img1.shape();
    assert_eq!(shape.len(), 4, "CutMix expects 4D batch [N, C, H, W]");
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

    let x1 = bbox[0].min(w);
    let y1 = bbox[1].min(h);
    let x2 = bbox[2].min(w);
    let y2 = bbox[3].min(h);

    let mut out_data = img1.to_vec();

    for b in 0..n {
        for ch in 0..c {
            for y in y1..y2 {
                for x in x1..x2 {
                    let idx = b * (c * h * w) + ch * (h * w) + y * w + x;
                    out_data[idx] = img2.get_4d(b, ch, y, x);
                }
            }
        }
    }

    Tensor::from_slice(&out_data, shape.to_vec())
}

/// Computes a randomized bounding box for CutMix with area proportional to `(1 - lambda)`.
pub fn sample_cutmix_box(img_w: usize, img_h: usize, lambda: f64) -> [usize; 4] {
    let cut_ratio = (1.0 - lambda.clamp(0.0, 1.0)).sqrt();
    let cut_w = ((img_w as f64) * cut_ratio) as usize;
    let cut_h = ((img_h as f64) * cut_ratio) as usize;

    let cx = img_w / 2;
    let cy = img_h / 2;

    let x1 = cx.saturating_sub(cut_w / 2);
    let y1 = cy.saturating_sub(cut_h / 2);
    let x2 = (x1 + cut_w).min(img_w);
    let y2 = (y1 + cut_h).min(img_h);

    [x1, y1, x2, y2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixup_and_cutmix() {
        let img1 = Tensor::zeros(vec![1, 1, 4, 4]);
        let img2 = Tensor::ones(vec![1, 1, 4, 4]);

        let mixed = mixup(&img1, &img2, 0.7);
        // 0.7 * 0 + 0.3 * 1 = 0.3
        assert!((mixed.data()[0] - 0.3).abs() < 1e-6);

        let cut = cutmix(&img1, &img2, &[1, 1, 3, 3]);
        assert_eq!(cut.shape(), &[1, 1, 4, 4]);
        // Pixel (0, 0) should be 0.0 (from img1)
        assert_eq!(cut.get_4d(0, 0, 0, 0), 0.0);
        // Pixel (1, 1) inside patch should be 1.0 (from img2)
        assert_eq!(cut.get_4d(0, 0, 1, 1), 1.0);
    }
}
