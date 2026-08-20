//! # Vectorized Bounding Box Operations
//!
//! High-performance bounding box area, intersection, union, and IoU/GIoU/DIoU/CIoU matrix calculations.

use brain_core::Tensor;

/// Computes areas for an array of bounding boxes `[N, 4]` (in XYXY format: x1, y1, x2, y2).
pub fn box_area(boxes: &Tensor) -> Tensor {
    let shape = boxes.shape();
    assert_eq!(shape.len(), 2, "boxes must be a 2D tensor [N, 4]");
    assert_eq!(
        shape[1], 4,
        "boxes must have 4 coordinates [x1, y1, x2, y2]"
    );

    let n = shape[0];
    let mut areas = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = boxes.get_2d(i, 0);
        let y1 = boxes.get_2d(i, 1);
        let x2 = boxes.get_2d(i, 2);
        let y2 = boxes.get_2d(i, 3);
        let w = (x2 - x1).max(0.0);
        let h = (y2 - y1).max(0.0);
        areas.push(w * h);
    }

    Tensor::from_slice(&areas, vec![n])
}

/// Computes pairwise IoU matrix between two sets of bounding boxes: `[N, 4]` and `[M, 4]`.
/// Returns an `[N, M]` IoU matrix where entry (i, j) is the IoU between boxes1[i] and boxes2[j].
pub fn box_iou_matrix(boxes1: &Tensor, boxes2: &Tensor) -> Tensor {
    let shape1 = boxes1.shape();
    let shape2 = boxes2.shape();
    assert_eq!(shape1.len(), 2, "boxes1 must be [N, 4]");
    assert_eq!(shape2.len(), 2, "boxes2 must be [M, 4]");
    assert_eq!(shape1[1], 4, "boxes1 must have 4 coordinates");
    assert_eq!(shape2[1], 4, "boxes2 must have 4 coordinates");

    let n = shape1[0];
    let m = shape2[0];
    let mut iou_matrix = vec![0.0f64; n * m];

    let area1 = box_area(boxes1);
    let area2 = box_area(boxes2);

    for i in 0..n {
        let a_x1 = boxes1.get_2d(i, 0);
        let a_y1 = boxes1.get_2d(i, 1);
        let a_x2 = boxes1.get_2d(i, 2);
        let a_y2 = boxes1.get_2d(i, 3);
        let a_area = area1.get(i);

        for j in 0..m {
            let b_x1 = boxes2.get_2d(j, 0);
            let b_y1 = boxes2.get_2d(j, 1);
            let b_x2 = boxes2.get_2d(j, 2);
            let b_y2 = boxes2.get_2d(j, 3);
            let b_area = area2.get(j);

            let inter_x1 = a_x1.max(b_x1);
            let inter_y1 = a_y1.max(b_y1);
            let inter_x2 = a_x2.min(b_x2);
            let inter_y2 = a_y2.min(b_y2);

            let inter_w = (inter_x2 - inter_x1).max(0.0);
            let inter_h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = inter_w * inter_h;

            let union_area = a_area + b_area - inter_area;
            let iou = if union_area > 0.0 {
                inter_area / union_area
            } else {
                0.0
            };

            iou_matrix[i * m + j] = iou;
        }
    }

    Tensor::from_slice(&iou_matrix, vec![n, m])
}

/// Computes Generalized Intersection over Union (GIoU) matrix.
pub fn box_giou_matrix(boxes1: &Tensor, boxes2: &Tensor) -> Tensor {
    let n = boxes1.shape()[0];
    let m = boxes2.shape()[0];
    let mut giou_matrix = vec![0.0f64; n * m];

    let area1 = box_area(boxes1);
    let area2 = box_area(boxes2);

    for i in 0..n {
        let a_x1 = boxes1.get_2d(i, 0);
        let a_y1 = boxes1.get_2d(i, 1);
        let a_x2 = boxes1.get_2d(i, 2);
        let a_y2 = boxes1.get_2d(i, 3);
        let a_area = area1.get(i);

        for j in 0..m {
            let b_x1 = boxes2.get_2d(j, 0);
            let b_y1 = boxes2.get_2d(j, 1);
            let b_x2 = boxes2.get_2d(j, 2);
            let b_y2 = boxes2.get_2d(j, 3);
            let b_area = area2.get(j);

            let inter_x1 = a_x1.max(b_x1);
            let inter_y1 = a_y1.max(b_y1);
            let inter_x2 = a_x2.min(b_x2);
            let inter_y2 = a_y2.min(b_y2);

            let inter_w = (inter_x2 - inter_x1).max(0.0);
            let inter_h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = inter_w * inter_h;

            let union_area = a_area + b_area - inter_area;
            let iou = if union_area > 0.0 {
                inter_area / union_area
            } else {
                0.0
            };

            // Smallest enclosing box
            let enc_x1 = a_x1.min(b_x1);
            let enc_y1 = a_y1.min(b_y1);
            let enc_x2 = a_x2.max(b_x2);
            let enc_y2 = a_y2.max(b_y2);

            let enc_w = (enc_x2 - enc_x1).max(0.0);
            let enc_h = (enc_y2 - enc_y1).max(0.0);
            let enc_area = enc_w * enc_h;

            let giou = if enc_area > 0.0 {
                iou - (enc_area - union_area) / enc_area
            } else {
                iou
            };

            giou_matrix[i * m + j] = giou;
        }
    }

    Tensor::from_slice(&giou_matrix, vec![n, m])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_area_and_iou() {
        let b1 = Tensor::from_slice(&[0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0], vec![2, 4]);
        let areas = box_area(&b1);
        assert_eq!(areas.data(), &[100.0, 100.0]);

        let b2 = Tensor::from_slice(&[0.0, 0.0, 10.0, 10.0], vec![1, 4]);
        let iou = box_iou_matrix(&b1, &b2);
        assert_eq!(iou.shape(), &[2, 1]);
        // Box 0 with Box 0 -> IoU = 1.0
        assert!((iou.data()[0] - 1.0).abs() < 1e-6);
        // Box 1 with Box 0 -> inter = 5x5 = 25, union = 100 + 100 - 25 = 175 -> IoU = 25/175 = 1/7 ~= 0.142857
        assert!((iou.data()[1] - (25.0 / 175.0)).abs() < 1e-5);
    }
}
