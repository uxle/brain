use brain_core::Tensor;
use brain_cv::{
    conv::{DepthwiseSeparableConv2d, GhostModule},
    detection::RoIAlign,
    feature::Fpn,
};

#[test]
fn test_depthwise_separable_conv2d() {
    let dw = DepthwiseSeparableConv2d::new(3, 16, 3);
    let x = Tensor::zeros(vec![1, 3, 32, 32]);
    let y = dw.forward(&x);
    assert_eq!(y.shape(), &[1, 16, 32, 32]);
}

#[test]
fn test_ghost_module() {
    let ghost = GhostModule::new(16, 32);
    let x = Tensor::zeros(vec![1, 16, 16, 16]);
    let y = ghost.forward(&x);
    assert_eq!(y.shape(), &[1, 32, 16, 16]);
}

#[test]
fn test_roi_align_spatial_pooling() {
    let roi = RoIAlign::new((7, 7), 0.25, 2);
    let features = Tensor::zeros(vec![1, 16, 32, 32]);
    let rois = Tensor::from_slice(&[0.0, 10.0, 10.0, 50.0, 50.0], vec![1, 5]);
    let pooled = roi.forward(&features, &rois);
    assert_eq!(pooled.shape(), &[1, 16, 7, 7]);
}

#[test]
fn test_fpn_pyramid_feature_shapes() {
    let fpn = Fpn::new(vec![64, 128, 256], 64);
    let c3 = Tensor::zeros(vec![1, 64, 32, 32]);
    let c4 = Tensor::zeros(vec![1, 128, 16, 16]);
    let c5 = Tensor::zeros(vec![1, 256, 8, 8]);
    let pyramid = fpn.forward(&[c3, c4, c5]);
    assert_eq!(pyramid.len(), 3);
    assert_eq!(pyramid[0].shape(), &[1, 64, 32, 32]);
    assert_eq!(pyramid[1].shape(), &[1, 64, 16, 16]);
    assert_eq!(pyramid[2].shape(), &[1, 64, 8, 8]);
}

#[test]
fn test_nms_and_box_iou_matrices() {
    use brain_cv::{box_area, box_giou_matrix, box_iou_matrix, non_max_suppression, NmsConfig};

    let boxes = Tensor::from_slice(
        &[
            0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 50.0, 50.0, 70.0, 70.0,
        ],
        vec![3, 4],
    );
    let scores = Tensor::from_slice(&[0.95, 0.85, 0.75], vec![3]);

    let areas = box_area(&boxes);
    assert_eq!(areas.data(), &[100.0, 81.0, 400.0]);

    let iou = box_iou_matrix(&boxes, &boxes);
    assert_eq!(iou.shape(), &[3, 3]);
    assert!((iou.get_2d(0, 0) - 1.0).abs() < 1e-6);

    let giou = box_giou_matrix(&boxes, &boxes);
    assert_eq!(giou.shape(), &[3, 3]);
    assert!((giou.get_2d(0, 0) - 1.0).abs() < 1e-6);

    let cfg = NmsConfig {
        iou_threshold: 0.5,
        score_threshold: 0.1,
        max_output_boxes: 10,
    };
    let kept = non_max_suppression(&boxes, &scores, &cfg);
    assert_eq!(kept, vec![0, 2]);
}

#[test]
fn test_mixup_and_cutmix_augmentations() {
    use brain_cv::{cutmix, mixup, sample_cutmix_box};

    let img1 = Tensor::zeros(vec![1, 3, 32, 32]);
    let img2 = Tensor::ones(vec![1, 3, 32, 32]);

    let mixed = mixup(&img1, &img2, 0.6);
    assert_eq!(mixed.shape(), &[1, 3, 32, 32]);
    assert!((mixed.data()[0] - 0.4).abs() < 1e-6);

    let bbox = sample_cutmix_box(32, 32, 0.5);
    let cut = cutmix(&img1, &img2, &bbox);
    assert_eq!(cut.shape(), &[1, 3, 32, 32]);
}
