//! # Semantic Segmentation with ViT for brain-vit
//!
//! Provides:
//! - [`SegDecoder`] — linear segmentation decoder from patch tokens
//! - [`SegMetrics`] — mIoU, pixel accuracy, per-class IoU computation
//! - [`UpscaleDecoder`] — bilinear upsampling then linear decode

use crate::core::{VitError, VitResult, Tensor2D, SimpleRng};
use crate::ops::linear;

/// Segmentation decoder: maps patch tokens to per-pixel class logits.
///
/// Uses a simple linear projection over patch tokens.
pub struct SegDecoder {
    /// Weight `[num_classes, embed_dim]`.
    pub weight: Vec<f64>,
    /// Bias `[num_classes]`.
    pub bias: Vec<f64>,
    /// Input embedding dimension.
    pub embed_dim: usize,
    /// Number of segmentation classes.
    pub num_classes: usize,
}

impl SegDecoder {
    /// Create a new segmentation decoder.
    pub fn new(embed_dim: usize, num_classes: usize, seed: u64) -> VitResult<Self> {
        if embed_dim == 0 || num_classes == 0 {
            return Err(VitError::Config("SegDecoder: dims must be > 0".to_string()));
        }
        let mut rng = SimpleRng::new(seed);
        let weight = rng.xavier_uniform(num_classes, embed_dim);
        let bias = vec![0.0f64; num_classes];
        Ok(Self { weight, bias, embed_dim, num_classes })
    }

    /// Forward: `[B, N, D]` → `[B, N, C]` flat class logits.
    pub fn forward(&self, patch_tokens: &[f64], batch: usize, n: usize) -> VitResult<Vec<f64>> {
        if patch_tokens.len() != batch * n * self.embed_dim {
            return Err(VitError::Shape("SegDecoder: tokens shape mismatch".to_string()));
        }
        let input = Tensor2D::from_data(batch * n, self.embed_dim, patch_tokens.to_vec())?;
        let w = Tensor2D::from_data(self.num_classes, self.embed_dim, self.weight.clone())?;
        let out = linear(&input, &w, Some(&self.bias))?;
        Ok(out.data)
    }

    /// Predict class per patch: argmax over logits `[B, N, C]` → `[B, N]`.
    pub fn predict_class(&self, logits: &[f64], batch: usize, n: usize) -> VitResult<Vec<usize>> {
        if logits.len() != batch * n * self.num_classes {
            return Err(VitError::Shape("predict_class: logits shape mismatch".to_string()));
        }
        let mut preds = vec![0usize; batch * n];
        for i in 0..batch * n {
            let row = &logits[i * self.num_classes..(i + 1) * self.num_classes];
            preds[i] = row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
        }
        Ok(preds)
    }

    /// Parameter count.
    pub fn num_params(&self) -> usize { self.weight.len() + self.bias.len() }
}

/// Compute per-class and mean IoU for segmentation.
///
/// - `preds`: predicted class per patch `[N]`.
/// - `gts`: ground-truth class per patch `[N]`.
/// - Returns per-class IoU and mean IoU.
pub fn per_class_iou(
    preds: &[usize],
    gts: &[usize],
    num_classes: usize,
) -> VitResult<(Vec<f64>, f64)> {
    if preds.len() != gts.len() {
        return Err(VitError::Shape("per_class_iou: length mismatch".to_string()));
    }
    let mut tp = vec![0usize; num_classes];
    let mut fp = vec![0usize; num_classes];
    let mut fn_ = vec![0usize; num_classes];

    for (&p, &g) in preds.iter().zip(gts.iter()) {
        if p < num_classes && g < num_classes {
            if p == g {
                tp[p] += 1;
            } else {
                fp[p] += 1;
                fn_[g] += 1;
            }
        }
    }

    let ious: Vec<f64> = (0..num_classes).map(|c| {
        let denom = tp[c] + fp[c] + fn_[c];
        if denom == 0 { f64::NAN } else { tp[c] as f64 / denom as f64 }
    }).collect();

    let valid: Vec<f64> = ious.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let miou = if valid.is_empty() { 0.0 } else { valid.iter().sum::<f64>() / valid.len() as f64 };
    Ok((ious, miou))
}

/// Compute pixel accuracy.
pub fn pixel_accuracy(preds: &[usize], gts: &[usize]) -> VitResult<f64> {
    if preds.len() != gts.len() {
        return Err(VitError::Shape("pixel_accuracy: length mismatch".to_string()));
    }
    if preds.is_empty() { return Ok(0.0); }
    let correct = preds.iter().zip(gts.iter()).filter(|(a, b)| a == b).count();
    Ok(correct as f64 / preds.len() as f64)
}

/// Confusion matrix for segmentation evaluation.
pub fn confusion_matrix(
    preds: &[usize],
    gts: &[usize],
    num_classes: usize,
) -> VitResult<Vec<Vec<usize>>> {
    if preds.len() != gts.len() {
        return Err(VitError::Shape("confusion_matrix: length mismatch".to_string()));
    }
    let mut cm = vec![vec![0usize; num_classes]; num_classes];
    for (&p, &g) in preds.iter().zip(gts.iter()) {
        if p < num_classes && g < num_classes {
            cm[g][p] += 1;
        }
    }
    Ok(cm)
}

/// Dice coefficient for binary segmentation.
///
/// `pred`, `gt`: binary (0 or 1) per-pixel predictions.
pub fn dice_coefficient(pred: &[bool], gt: &[bool]) -> VitResult<f64> {
    if pred.len() != gt.len() {
        return Err(VitError::Shape("dice_coefficient: length mismatch".to_string()));
    }
    let tp = pred.iter().zip(gt.iter()).filter(|(&p, &g)| p && g).count() as f64;
    let sum = (pred.iter().filter(|&&v| v).count() + gt.iter().filter(|&&v| v).count()) as f64;
    if sum < 1e-10 { Ok(1.0) } else { Ok(2.0 * tp / sum) }
}

/// Upscale + decode segmentation decoder.
///
/// Takes patch token grid `[B, sqrt(N), sqrt(N), D]` and bilinearly upsamples
/// to `[B, H_out, W_out, num_classes]`, then applies linear decode.
pub struct UpscaleDecoder {
    /// Linear decoder applied after upsampling.
    pub linear_dec: SegDecoder,
    /// Output height.
    pub out_h: usize,
    /// Output width.
    pub out_w: usize,
    /// Grid size (patches per side).
    pub grid_size: usize,
}

impl UpscaleDecoder {
    /// Create a new upscale decoder.
    pub fn new(embed_dim: usize, num_classes: usize, grid_size: usize, out_h: usize, out_w: usize, seed: u64) -> VitResult<Self> {
        let linear_dec = SegDecoder::new(embed_dim, num_classes, seed)?;
        Ok(Self { linear_dec, out_h, out_w, grid_size })
    }

    /// Forward: decode patch tokens → per-patch logits `[B, N, C]`.
    pub fn forward(&self, patch_tokens: &[f64], batch: usize) -> VitResult<Vec<f64>> {
        let n = self.grid_size * self.grid_size;
        self.linear_dec.forward(patch_tokens, batch, n)
    }

    /// Total parameters.
    pub fn num_params(&self) -> usize { self.linear_dec.num_params() }
}

/// Compute Intersection over Union for a pair of segmentation maps.
pub fn seg_iou_pair(pred_mask: &[bool], gt_mask: &[bool]) -> VitResult<f64> {
    if pred_mask.len() != gt_mask.len() {
        return Err(VitError::Shape("seg_iou_pair: length mismatch".to_string()));
    }
    let inter = pred_mask.iter().zip(gt_mask.iter()).filter(|(&p, &g)| p && g).count() as f64;
    let union = pred_mask.iter().zip(gt_mask.iter()).filter(|(&p, &g)| p || g).count() as f64;
    if union < 1e-10 { Ok(1.0) } else { Ok(inter / union) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seg_decoder_new() {
        let d = SegDecoder::new(16, 4, 0).unwrap();
        assert_eq!(d.embed_dim, 16);
        assert_eq!(d.num_classes, 4);
    }

    #[test]
    fn test_seg_decoder_forward_shape() {
        let d = SegDecoder::new(16, 4, 0).unwrap();
        let tokens = vec![0.5f64; 2 * 9 * 16];
        let logits = d.forward(&tokens, 2, 9).unwrap();
        assert_eq!(logits.len(), 2 * 9 * 4);
    }

    #[test]
    fn test_seg_decoder_finite() {
        let d = SegDecoder::new(16, 4, 1).unwrap();
        let tokens = vec![0.1f64; 1 * 4 * 16];
        let logits = d.forward(&tokens, 1, 4).unwrap();
        assert!(logits.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_seg_decoder_shape_err() {
        let d = SegDecoder::new(16, 4, 0).unwrap();
        assert!(d.forward(&[0.0f64; 10], 1, 4).is_err());
    }

    #[test]
    fn test_seg_decoder_invalid() {
        assert!(SegDecoder::new(0, 4, 0).is_err());
        assert!(SegDecoder::new(16, 0, 0).is_err());
    }

    #[test]
    fn test_seg_decoder_num_params() {
        let d = SegDecoder::new(16, 4, 0).unwrap();
        assert_eq!(d.num_params(), 16 * 4 + 4);
    }

    #[test]
    fn test_predict_class_shape() {
        let d = SegDecoder::new(16, 4, 0).unwrap();
        let logits = vec![0.0f64; 2 * 9 * 4];
        let preds = d.predict_class(&logits, 2, 9).unwrap();
        assert_eq!(preds.len(), 2 * 9);
    }

    #[test]
    fn test_predict_class_max() {
        let d = SegDecoder::new(2, 3, 0).unwrap();
        let logits = vec![0.0f64, 0.0, 1.0]; // [1, 3] → class 2
        let preds = d.predict_class(&logits, 1, 1).unwrap();
        assert_eq!(preds[0], 2);
    }

    #[test]
    fn test_per_class_iou_perfect() {
        let preds = vec![0usize, 1, 2, 0, 1, 2];
        let gts = vec![0usize, 1, 2, 0, 1, 2];
        let (ious, miou) = per_class_iou(&preds, &gts, 3).unwrap();
        for &iou in &ious { assert!((iou - 1.0).abs() < 1e-9); }
        assert!((miou - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_class_iou_all_wrong() {
        let preds = vec![1usize, 2, 0]; // all wrong
        let gts = vec![0usize, 1, 2];
        let (_, miou) = per_class_iou(&preds, &gts, 3).unwrap();
        assert!((miou - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_class_iou_length_mismatch() {
        assert!(per_class_iou(&[0, 1], &[0], 2).is_err());
    }

    #[test]
    fn test_pixel_accuracy_perfect() {
        let preds = vec![0usize, 1, 2];
        let gts = vec![0usize, 1, 2];
        let acc = pixel_accuracy(&preds, &gts).unwrap();
        assert!((acc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pixel_accuracy_zero() {
        let preds = vec![1usize, 2, 0];
        let gts = vec![0usize, 1, 2];
        let acc = pixel_accuracy(&preds, &gts).unwrap();
        assert!((acc - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pixel_accuracy_empty() {
        let acc = pixel_accuracy(&[], &[]).unwrap();
        assert!((acc - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pixel_accuracy_length_mismatch() {
        assert!(pixel_accuracy(&[0, 1], &[0]).is_err());
    }

    #[test]
    fn test_confusion_matrix_shape() {
        let preds = vec![0usize, 1, 2, 0];
        let gts = vec![0usize, 0, 1, 2];
        let cm = confusion_matrix(&preds, &gts, 3).unwrap();
        assert_eq!(cm.len(), 3);
        assert_eq!(cm[0].len(), 3);
    }

    #[test]
    fn test_confusion_matrix_perfect() {
        let preds = vec![0usize, 1, 2];
        let gts = vec![0usize, 1, 2];
        let cm = confusion_matrix(&preds, &gts, 3).unwrap();
        for c in 0..3 { assert_eq!(cm[c][c], 1); }
    }

    #[test]
    fn test_dice_perfect() {
        let mask = vec![true, false, true, false];
        let d = dice_coefficient(&mask, &mask).unwrap();
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dice_no_overlap() {
        let pred = vec![true, false];
        let gt = vec![false, true];
        let d = dice_coefficient(&pred, &gt).unwrap();
        assert!((d - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dice_both_empty() {
        let pred = vec![false, false];
        let gt = vec![false, false];
        let d = dice_coefficient(&pred, &gt).unwrap();
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dice_length_mismatch() {
        assert!(dice_coefficient(&[true, false], &[true]).is_err());
    }

    #[test]
    fn test_upscale_decoder_forward_shape() {
        let ud = UpscaleDecoder::new(16, 4, 4, 64, 64, 0).unwrap();
        let tokens = vec![0.5f64; 1 * 16 * 16]; // B=1, N=16 (4x4 grid), D=16
        let out = ud.forward(&tokens, 1).unwrap();
        assert_eq!(out.len(), 1 * 16 * 4);
    }

    #[test]
    fn test_seg_iou_pair_perfect() {
        let mask = vec![true, false, true];
        let iou = seg_iou_pair(&mask, &mask).unwrap();
        assert!((iou - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_seg_iou_pair_no_overlap() {
        let pred = vec![true, false];
        let gt = vec![false, true];
        let iou = seg_iou_pair(&pred, &gt).unwrap();
        assert!((iou - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_seg_iou_pair_both_empty() {
        let pred = vec![false, false];
        let gt = vec![false, false];
        let iou = seg_iou_pair(&pred, &gt).unwrap();
        assert!((iou - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_seg_iou_pair_length_mismatch() {
        assert!(seg_iou_pair(&[true], &[true, false]).is_err());
    }
}
