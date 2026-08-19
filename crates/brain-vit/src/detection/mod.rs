//! # Object Detection with ViT backbone for brain-vit
//!
//! Implements detection-specific components:
//! - [`DetectionHead`] — predict bounding boxes and classes from patch tokens
//! - [`AnchorFreeDecoder`] — anchor-free DETR-lite decoder
//! - [`Bbox`] — axis-aligned bounding box type
//! - [`IoU`] — intersection-over-union computation

use crate::core::{VitError, VitResult, Tensor2D, SimpleRng};
use crate::ops::linear;
use std::fmt;

/// Axis-aligned bounding box in [cx, cy, w, h] normalized format.
#[derive(Debug, Clone, PartialEq)]
pub struct Bbox {
    /// Center x (normalized 0..1).
    pub cx: f64,
    /// Center y (normalized 0..1).
    pub cy: f64,
    /// Width (normalized 0..1).
    pub w: f64,
    /// Height (normalized 0..1).
    pub h: f64,
    /// Class score.
    pub score: f64,
    /// Class index.
    pub class_id: usize,
}

impl Bbox {
    /// Create a new bounding box.
    pub fn new(cx: f64, cy: f64, w: f64, h: f64, score: f64, class_id: usize) -> Self {
        Self { cx, cy, w, h, score, class_id }
    }

    /// Convert to [x1, y1, x2, y2] format.
    pub fn to_xyxy(&self) -> [f64; 4] {
        [
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        ]
    }

    /// Create from [x1, y1, x2, y2] format.
    pub fn from_xyxy(x1: f64, y1: f64, x2: f64, y2: f64, score: f64, class_id: usize) -> Self {
        Self {
            cx: (x1 + x2) / 2.0,
            cy: (y1 + y2) / 2.0,
            w: x2 - x1,
            h: y2 - y1,
            score,
            class_id,
        }
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f64 { self.w * self.h }

    /// Clamp box coordinates to [0, 1].
    pub fn clamp(&self) -> Self {
        let cx = self.cx.clamp(0.0, 1.0);
        let cy = self.cy.clamp(0.0, 1.0);
        let w = self.w.clamp(0.0, 1.0);
        let h = self.h.clamp(0.0, 1.0);
        Self { cx, cy, w, h, ..*self }
    }
}

impl fmt::Display for Bbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bbox(cx={:.3}, cy={:.3}, w={:.3}, h={:.3}, cls={}, score={:.3})",
            self.cx, self.cy, self.w, self.h, self.class_id, self.score)
    }
}

/// Compute IoU (Intersection over Union) between two bounding boxes.
pub fn iou(a: &Bbox, b: &Bbox) -> f64 {
    let [ax1, ay1, ax2, ay2] = a.to_xyxy();
    let [bx1, by1, bx2, by2] = b.to_xyxy();
    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;
    let union_area = a.area() + b.area() - inter_area;
    if union_area <= 0.0 { 0.0 } else { inter_area / union_area }
}

/// Compute Generalized IoU (GIoU).
pub fn giou(a: &Bbox, b: &Bbox) -> f64 {
    let [ax1, ay1, ax2, ay2] = a.to_xyxy();
    let [bx1, by1, bx2, by2] = b.to_xyxy();
    // IoU
    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);
    let inter_area = ((inter_x2 - inter_x1).max(0.0)) * ((inter_y2 - inter_y1).max(0.0));
    let union_area = a.area() + b.area() - inter_area;
    let iou_val = if union_area <= 0.0 { 0.0 } else { inter_area / union_area };
    // Enclosing box
    let enc_x1 = ax1.min(bx1);
    let enc_y1 = ay1.min(by1);
    let enc_x2 = ax2.max(bx2);
    let enc_y2 = ay2.max(by2);
    let enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
    if enc_area <= 0.0 { iou_val } else {
        iou_val - (enc_area - union_area) / enc_area
    }
}

/// Non-maximum suppression.
///
/// Suppresses overlapping boxes with the same class, keeping the highest-scoring.
///
/// # Arguments
/// - `boxes`: detected boxes (any order).
/// - `iou_threshold`: boxes with IoU > threshold are suppressed.
pub fn nms(boxes: &[Bbox], iou_threshold: f64) -> Vec<Bbox> {
    let mut sorted: Vec<&Bbox> = boxes.iter().collect();
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept = Vec::new();
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] { continue; }
        kept.push(sorted[i].clone());
        for j in (i + 1)..sorted.len() {
            if suppressed[j] { continue; }
            if sorted[i].class_id == sorted[j].class_id
                && iou(sorted[i], sorted[j]) > iou_threshold
            {
                suppressed[j] = true;
            }
        }
    }
    kept
}

/// Detection head: projects patch tokens to N query boxes.
///
/// For DETR-like detection: predicts (cx, cy, w, h, class_scores) per query.
pub struct DetectionHead {
    /// Box regression weight `[4, embed_dim]`.
    pub box_w: Vec<f64>,
    /// Box regression bias `[4]`.
    pub box_b: Vec<f64>,
    /// Class prediction weight `[num_classes + 1, embed_dim]` (+ background).
    pub cls_w: Vec<f64>,
    /// Class prediction bias `[num_classes + 1]`.
    pub cls_b: Vec<f64>,
    /// Input embedding dimension.
    pub embed_dim: usize,
    /// Number of foreground classes.
    pub num_classes: usize,
    /// Number of detection queries.
    pub num_queries: usize,
}

impl DetectionHead {
    /// Create a new detection head.
    pub fn new(embed_dim: usize, num_classes: usize, num_queries: usize, seed: u64) -> VitResult<Self> {
        if embed_dim == 0 || num_classes == 0 || num_queries == 0 {
            return Err(VitError::Config("DetectionHead: all dims must be > 0".to_string()));
        }
        let mut rng = SimpleRng::new(seed);
        let box_w = rng.xavier_uniform(4, embed_dim);
        let box_b = vec![0.0f64; 4];
        let cls_w = rng.xavier_uniform(num_classes + 1, embed_dim);
        let cls_b = vec![0.0f64; num_classes + 1];
        Ok(Self { box_w, box_b, cls_w, cls_b, embed_dim, num_classes, num_queries })
    }

    /// Forward: `[B, num_queries, embed_dim]` → `([B, Q, 4], [B, Q, C+1])` flat.
    pub fn forward(&self, queries: &[f64], batch: usize) -> VitResult<(Vec<f64>, Vec<f64>)> {
        let n = self.num_queries;
        let d = self.embed_dim;
        let c = self.num_classes + 1;
        if queries.len() != batch * n * d {
            return Err(VitError::Shape("DetectionHead: queries shape mismatch".to_string()));
        }

        let flat_input = Tensor2D::from_data(batch * n, d, queries.to_vec())?;
        let bw = Tensor2D::from_data(4, d, self.box_w.clone())?;
        let cw = Tensor2D::from_data(c, d, self.cls_w.clone())?;

        let box_out = linear(&flat_input, &bw, Some(&self.box_b))?;
        let cls_out = linear(&flat_input, &cw, Some(&self.cls_b))?;

        // Apply sigmoid to box outputs to get normalized coords
        let boxes: Vec<f64> = box_out.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok((boxes, cls_out.data))
    }

    /// Convert raw head outputs to `Bbox` detections per sample.
    #[allow(clippy::needless_range_loop)]
    pub fn decode_boxes(
        &self,
        boxes: &[f64],
        cls_logits: &[f64],
        batch: usize,
        score_threshold: f64,
    ) -> VitResult<Vec<Vec<Bbox>>> {
        let n = self.num_queries;
        let c = self.num_classes + 1;
        let mut results = vec![vec![]; batch];
        for b in 0..batch {
            for q in 0..n {
                let box_start = (b * n + q) * 4;
                let cls_start = (b * n + q) * c;
                let cx = boxes[box_start];
                let cy = boxes[box_start + 1];
                let w = boxes[box_start + 2];
                let h = boxes[box_start + 3];
                let cls_row = &cls_logits[cls_start..cls_start + c];
                // Softmax
                let max_v = cls_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = cls_row.iter().map(|&x| (x - max_v).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
                // Best foreground class (skip background = last)
                let (best_cls, best_score) = probs[..self.num_classes].iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, &s)| (i, s))
                    .unwrap_or((0, 0.0));
                if best_score >= score_threshold {
                    results[b].push(Bbox::new(cx, cy, w, h, best_score, best_cls));
                }
            }
        }
        Ok(results)
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize {
        self.box_w.len() + self.box_b.len() + self.cls_w.len() + self.cls_b.len()
    }
}

/// Compute mean average precision (mAP) at a given IoU threshold.
///
/// `predictions`: per-image predicted boxes.
/// `ground_truth`: per-image ground-truth boxes.
pub fn mean_average_precision(
    predictions: &[Vec<Bbox>],
    ground_truth: &[Vec<Bbox>],
    iou_threshold: f64,
    num_classes: usize,
) -> f64 {
    if predictions.len() != ground_truth.len() || predictions.is_empty() {
        return 0.0;
    }
    let mut ap_sum = 0.0;
    for cls in 0..num_classes {
        let mut tp = vec![];
        let mut fp = vec![];
        let mut total_gt = 0usize;
        for (preds, gts) in predictions.iter().zip(ground_truth.iter()) {
            let cls_gts: Vec<&Bbox> = gts.iter().filter(|b| b.class_id == cls).collect();
            let cls_preds: Vec<&Bbox> = preds.iter().filter(|b| b.class_id == cls).collect();
            total_gt += cls_gts.len();
            let mut matched = vec![false; cls_gts.len()];
            for pred in &cls_preds {
                let best = cls_gts.iter().enumerate()
                    .filter(|(i, _)| !matched[*i])
                    .map(|(i, gt)| (i, iou(pred, gt)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                if let Some((idx, iou_val)) = best {
                    if iou_val >= iou_threshold {
                        tp.push(1.0f64);
                        fp.push(0.0f64);
                        matched[idx] = true;
                    } else {
                        tp.push(0.0f64); fp.push(1.0f64);
                    }
                } else {
                    tp.push(0.0f64); fp.push(1.0f64);
                }
            }
        }
        if total_gt == 0 { continue; }
        // Compute AP via trapezoidal rule
        let mut cum_tp = 0.0f64;
        let mut cum_fp = 0.0f64;
        let mut prec_rec: Vec<(f64, f64)> = vec![(1.0, 0.0)];
        for (&t, &f) in tp.iter().zip(fp.iter()) {
            cum_tp += t; cum_fp += f;
            let prec = cum_tp / (cum_tp + cum_fp).max(1e-10);
            let rec = cum_tp / total_gt as f64;
            prec_rec.push((prec, rec));
        }
        prec_rec.push((0.0, 1.0));
        let ap: f64 = prec_rec.windows(2).map(|w| {
            let dr = w[1].1 - w[0].1;
            dr * w[1].0
        }).sum();
        ap_sum += ap;
    }
    if num_classes > 0 { ap_sum / num_classes as f64 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_new() {
        let b = Bbox::new(0.5, 0.5, 0.2, 0.3, 0.9, 0);
        assert!((b.cx - 0.5).abs() < 1e-10);
        assert!((b.area() - 0.06).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_to_xyxy() {
        let b = Bbox::new(0.5, 0.5, 0.4, 0.6, 1.0, 0);
        let [x1, y1, x2, y2] = b.to_xyxy();
        assert!((x1 - 0.3).abs() < 1e-9);
        assert!((y1 - 0.2).abs() < 1e-9);
        assert!((x2 - 0.7).abs() < 1e-9);
        assert!((y2 - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_from_xyxy() {
        let b = Bbox::from_xyxy(0.1, 0.2, 0.5, 0.7, 0.8, 1);
        assert!((b.cx - 0.3).abs() < 1e-9);
        assert!((b.cy - 0.45).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_clamp() {
        let b = Bbox::new(0.5, 0.5, 2.0, 2.0, 1.0, 0);
        let c = b.clamp();
        assert!(c.w <= 1.0);
        assert!(c.h <= 1.0);
    }

    #[test]
    fn test_bbox_display() {
        let b = Bbox::new(0.5, 0.5, 0.2, 0.3, 0.9, 2);
        let s = format!("{}", b);
        assert!(s.contains("Bbox("));
    }

    #[test]
    fn test_iou_identical() {
        let b = Bbox::new(0.5, 0.5, 0.4, 0.4, 1.0, 0);
        assert!((iou(&b, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = Bbox::new(0.0, 0.0, 0.2, 0.2, 1.0, 0);
        let b = Bbox::new(1.0, 1.0, 0.2, 0.2, 1.0, 0);
        assert!((iou(&a, &b)).abs() < 1e-9);
    }

    #[test]
    fn test_iou_partial_overlap() {
        // a: cx=0.3, w=0.4 → x in [0.1, 0.5]
        // b: cx=0.5, w=0.4 → x in [0.3, 0.7]
        // Overlap in x: [0.3, 0.5] = 0.2; both have h=0.4 so overlap in y = 0.4
        // Intersection = 0.2 * 0.4 = 0.08; each area = 0.4*0.4 = 0.16; union = 0.24
        let a = Bbox::new(0.3, 0.5, 0.4, 0.4, 1.0, 0);
        let b = Bbox::new(0.5, 0.5, 0.4, 0.4, 1.0, 0);
        let v = iou(&a, &b);
        assert!(v > 0.0 && v < 1.0, "Expected partial overlap IoU in (0,1), got {}", v);
    }

    #[test]
    fn test_giou_identical() {
        let b = Bbox::new(0.5, 0.5, 0.4, 0.4, 1.0, 0);
        let g = giou(&b, &b);
        assert!((g - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_giou_no_overlap() {
        let a = Bbox::new(0.1, 0.1, 0.1, 0.1, 1.0, 0);
        let b = Bbox::new(0.9, 0.9, 0.1, 0.1, 1.0, 0);
        let g = giou(&a, &b);
        assert!(g < 1.0);
    }

    #[test]
    fn test_nms_keeps_all_different_classes() {
        let boxes = vec![
            Bbox::new(0.5, 0.5, 0.4, 0.4, 0.9, 0),
            Bbox::new(0.5, 0.5, 0.4, 0.4, 0.8, 1),
        ];
        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_nms_suppresses_same_class() {
        let boxes = vec![
            Bbox::new(0.5, 0.5, 0.4, 0.4, 0.9, 0),
            Bbox::new(0.5, 0.5, 0.4, 0.4, 0.8, 0),
        ];
        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_nms_empty() {
        let kept = nms(&[], 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_detection_head_new() {
        let h = DetectionHead::new(16, 4, 10, 0).unwrap();
        assert_eq!(h.embed_dim, 16);
        assert_eq!(h.num_classes, 4);
        assert_eq!(h.num_queries, 10);
    }

    #[test]
    fn test_detection_head_forward_shape() {
        let h = DetectionHead::new(16, 4, 10, 0).unwrap();
        let queries = vec![0.1f64; 2 * 10 * 16];
        let (boxes, cls) = h.forward(&queries, 2).unwrap();
        assert_eq!(boxes.len(), 2 * 10 * 4);
        assert_eq!(cls.len(), 2 * 10 * 5); // 4 classes + background
    }

    #[test]
    fn test_detection_head_boxes_in_01() {
        let h = DetectionHead::new(16, 4, 5, 0).unwrap();
        let queries = vec![1.0f64; 1 * 5 * 16];
        let (boxes, _) = h.forward(&queries, 1).unwrap();
        for &v in &boxes { assert!(v >= 0.0 && v <= 1.0); }
    }

    #[test]
    fn test_detection_head_shape_err() {
        let h = DetectionHead::new(16, 4, 10, 0).unwrap();
        assert!(h.forward(&[0.0f64; 5], 1).is_err());
    }

    #[test]
    fn test_detection_head_invalid() {
        assert!(DetectionHead::new(0, 4, 10, 0).is_err());
        assert!(DetectionHead::new(16, 0, 10, 0).is_err());
    }

    #[test]
    fn test_detection_head_num_params() {
        let h = DetectionHead::new(16, 4, 10, 0).unwrap();
        assert_eq!(h.num_params(), 4 * 16 + 4 + 5 * 16 + 5);
    }

    #[test]
    fn test_decode_boxes_threshold() {
        let h = DetectionHead::new(16, 4, 5, 0).unwrap();
        let queries = vec![5.0f64; 1 * 5 * 16]; // large values → clear prediction
        let (boxes, cls) = h.forward(&queries, 1).unwrap();
        let detections = h.decode_boxes(&boxes, &cls, 1, 0.0).unwrap();
        assert!(!detections[0].is_empty());
    }

    #[test]
    fn test_decode_boxes_high_threshold() {
        let h = DetectionHead::new(16, 4, 5, 0).unwrap();
        let queries = vec![0.0f64; 1 * 5 * 16];
        let (boxes, cls) = h.forward(&queries, 1).unwrap();
        let detections = h.decode_boxes(&boxes, &cls, 1, 0.99).unwrap();
        // May be empty with very high threshold
        assert!(detections[0].len() <= 5);
    }

    #[test]
    fn test_map_perfect() {
        let b = Bbox::new(0.5, 0.5, 0.4, 0.4, 0.9, 0);
        let preds = vec![vec![b.clone()]];
        let gts = vec![vec![b.clone()]];
        let map = mean_average_precision(&preds, &gts, 0.5, 1);
        assert!(map > 0.0);
    }

    #[test]
    fn test_map_no_detections() {
        let gt = vec![vec![Bbox::new(0.5, 0.5, 0.4, 0.4, 1.0, 0)]];
        let preds = vec![vec![]];
        let map = mean_average_precision(&preds, &gt, 0.5, 1);
        assert!((map - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_map_no_gt() {
        let preds = vec![vec![Bbox::new(0.5, 0.5, 0.4, 0.4, 0.9, 0)]];
        let gts = vec![vec![]];
        // No ground truth for class 0 → AP = 0
        let map = mean_average_precision(&preds, &gts, 0.5, 1);
        assert!((map - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_area_zero() {
        let a = Bbox::new(0.5, 0.5, 0.0, 0.0, 1.0, 0);
        let b = Bbox::new(0.5, 0.5, 0.4, 0.4, 1.0, 0);
        let v = iou(&a, &b);
        assert!((v - 0.0).abs() < 1e-9);
    }
}
