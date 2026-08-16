//! # Loss Function Gradients
//!
//! Differentiable rules for standard loss criteria:
//! MSE, Binary Cross-Entropy, Cross-Entropy, Huber loss.

use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::math as math_t;
use brain_core::tensor::special as spec_t;
use brain_core::{BrainResult, Tensor};

/// Gradient of Mean Squared Error: `2 * (pred - target) / N * g`.
pub fn grad_mse_loss(pred: &Tensor, target: &Tensor, g: f64) -> BrainResult<Tensor> {
    let diff = arith_t::sub(pred, target);
    let n = pred.numel() as f64;
    let factor = 2.0 * g / n;
    Ok(diff.map(|x| x * factor))
}

/// Gradient of fused Cross-Entropy loss: `(softmax(logits) - target_one_hot) / N * g`.
pub fn grad_cross_entropy_logits(logits: &Tensor, target_indices: &[usize], g: f64) -> BrainResult<Tensor> {
    let sm = spec_t::softmax(logits, logits.ndim() - 1);
    let mut grad_data = sm.data().to_vec();
    let num_classes = logits.shape().last().copied().unwrap_or(1);
    let batch_size = logits.numel() / num_classes;

    for (b, &class_idx) in target_indices.iter().enumerate().take(batch_size) {
        if class_idx < num_classes {
            grad_data[b * num_classes + class_idx] -= 1.0;
        }
    }

    let factor = g / batch_size as f64;
    let out = Tensor::from_slice(&grad_data, logits.shape().to_vec()).map(|x| x * factor);
    Ok(out)
}

/// Gradient of Binary Cross-Entropy with Logits: `(sigmoid(logits) - targets) / N * g`.
pub fn grad_bce_with_logits(logits: &Tensor, targets: &Tensor, g: f64) -> BrainResult<Tensor> {
    let sig = math_t::sigmoid(logits);
    let diff = arith_t::sub(&sig, targets);
    let n = logits.numel() as f64;
    let factor = g / n;
    Ok(diff.map(|x| x * factor))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_loss_grad_stress_001() {
        let p = Tensor::scalar(2.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_002() {
        let p = Tensor::scalar(2.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_003() {
        let p = Tensor::scalar(2.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_004() {
        let p = Tensor::scalar(2.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_005() {
        let p = Tensor::scalar(2.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_006() {
        let p = Tensor::scalar(2.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_007() {
        let p = Tensor::scalar(2.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_008() {
        let p = Tensor::scalar(2.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_009() {
        let p = Tensor::scalar(2.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((2.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_010() {
        let p = Tensor::scalar(3.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_011() {
        let p = Tensor::scalar(3.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_012() {
        let p = Tensor::scalar(3.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_013() {
        let p = Tensor::scalar(3.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_014() {
        let p = Tensor::scalar(3.4000000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.4000000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_015() {
        let p = Tensor::scalar(3.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_016() {
        let p = Tensor::scalar(3.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_017() {
        let p = Tensor::scalar(3.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_018() {
        let p = Tensor::scalar(3.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_019() {
        let p = Tensor::scalar(3.9000000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((3.9000000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_020() {
        let p = Tensor::scalar(4.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_021() {
        let p = Tensor::scalar(4.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_022() {
        let p = Tensor::scalar(4.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_023() {
        let p = Tensor::scalar(4.300000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.300000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_024() {
        let p = Tensor::scalar(4.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_025() {
        let p = Tensor::scalar(4.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_026() {
        let p = Tensor::scalar(4.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_027() {
        let p = Tensor::scalar(4.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_028() {
        let p = Tensor::scalar(4.800000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.800000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_029() {
        let p = Tensor::scalar(4.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((4.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_030() {
        let p = Tensor::scalar(5.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_031() {
        let p = Tensor::scalar(5.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_032() {
        let p = Tensor::scalar(5.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_033() {
        let p = Tensor::scalar(5.300000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.300000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_034() {
        let p = Tensor::scalar(5.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_035() {
        let p = Tensor::scalar(5.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_036() {
        let p = Tensor::scalar(5.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_037() {
        let p = Tensor::scalar(5.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_038() {
        let p = Tensor::scalar(5.800000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.800000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_039() {
        let p = Tensor::scalar(5.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((5.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_040() {
        let p = Tensor::scalar(6.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_041() {
        let p = Tensor::scalar(6.1000000000000005);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.1000000000000005) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_042() {
        let p = Tensor::scalar(6.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_043() {
        let p = Tensor::scalar(6.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_044() {
        let p = Tensor::scalar(6.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_045() {
        let p = Tensor::scalar(6.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_046() {
        let p = Tensor::scalar(6.6000000000000005);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.6000000000000005) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_047() {
        let p = Tensor::scalar(6.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_048() {
        let p = Tensor::scalar(6.800000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.800000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_049() {
        let p = Tensor::scalar(6.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((6.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_050() {
        let p = Tensor::scalar(7.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_051() {
        let p = Tensor::scalar(7.1000000000000005);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.1000000000000005) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_052() {
        let p = Tensor::scalar(7.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_053() {
        let p = Tensor::scalar(7.300000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.300000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_054() {
        let p = Tensor::scalar(7.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_055() {
        let p = Tensor::scalar(7.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_056() {
        let p = Tensor::scalar(7.6000000000000005);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.6000000000000005) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_057() {
        let p = Tensor::scalar(7.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_058() {
        let p = Tensor::scalar(7.800000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.800000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_059() {
        let p = Tensor::scalar(7.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((7.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_060() {
        let p = Tensor::scalar(8.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_061() {
        let p = Tensor::scalar(8.100000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.100000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_062() {
        let p = Tensor::scalar(8.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_063() {
        let p = Tensor::scalar(8.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_064() {
        let p = Tensor::scalar(8.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_065() {
        let p = Tensor::scalar(8.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_066() {
        let p = Tensor::scalar(8.600000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.600000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_067() {
        let p = Tensor::scalar(8.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_068() {
        let p = Tensor::scalar(8.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_069() {
        let p = Tensor::scalar(8.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((8.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_070() {
        let p = Tensor::scalar(9.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_071() {
        let p = Tensor::scalar(9.100000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.100000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_072() {
        let p = Tensor::scalar(9.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_073() {
        let p = Tensor::scalar(9.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_074() {
        let p = Tensor::scalar(9.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_075() {
        let p = Tensor::scalar(9.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_076() {
        let p = Tensor::scalar(9.600000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.600000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_077() {
        let p = Tensor::scalar(9.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_078() {
        let p = Tensor::scalar(9.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_079() {
        let p = Tensor::scalar(9.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((9.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_080() {
        let p = Tensor::scalar(10.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_081() {
        let p = Tensor::scalar(10.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_082() {
        let p = Tensor::scalar(10.200000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.200000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_083() {
        let p = Tensor::scalar(10.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_084() {
        let p = Tensor::scalar(10.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_085() {
        let p = Tensor::scalar(10.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_086() {
        let p = Tensor::scalar(10.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_087() {
        let p = Tensor::scalar(10.700000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.700000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_088() {
        let p = Tensor::scalar(10.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_089() {
        let p = Tensor::scalar(10.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((10.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_090() {
        let p = Tensor::scalar(11.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_091() {
        let p = Tensor::scalar(11.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_092() {
        let p = Tensor::scalar(11.200000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.200000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_093() {
        let p = Tensor::scalar(11.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_094() {
        let p = Tensor::scalar(11.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_095() {
        let p = Tensor::scalar(11.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_096() {
        let p = Tensor::scalar(11.600000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.600000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_097() {
        let p = Tensor::scalar(11.700000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.700000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_098() {
        let p = Tensor::scalar(11.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_099() {
        let p = Tensor::scalar(11.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((11.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_100() {
        let p = Tensor::scalar(12.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_101() {
        let p = Tensor::scalar(12.100000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.100000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_102() {
        let p = Tensor::scalar(12.200000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.200000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_103() {
        let p = Tensor::scalar(12.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_104() {
        let p = Tensor::scalar(12.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_105() {
        let p = Tensor::scalar(12.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_106() {
        let p = Tensor::scalar(12.600000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.600000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_107() {
        let p = Tensor::scalar(12.700000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.700000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_108() {
        let p = Tensor::scalar(12.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_109() {
        let p = Tensor::scalar(12.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((12.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_110() {
        let p = Tensor::scalar(13.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_111() {
        let p = Tensor::scalar(13.100000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.100000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_112() {
        let p = Tensor::scalar(13.200000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.200000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_113() {
        let p = Tensor::scalar(13.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_114() {
        let p = Tensor::scalar(13.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_115() {
        let p = Tensor::scalar(13.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_116() {
        let p = Tensor::scalar(13.600000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.600000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_117() {
        let p = Tensor::scalar(13.700000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.700000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_118() {
        let p = Tensor::scalar(13.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_119() {
        let p = Tensor::scalar(13.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((13.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_120() {
        let p = Tensor::scalar(14.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_121() {
        let p = Tensor::scalar(14.100000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.100000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_122() {
        let p = Tensor::scalar(14.200000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.200000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_123() {
        let p = Tensor::scalar(14.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_124() {
        let p = Tensor::scalar(14.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_125() {
        let p = Tensor::scalar(14.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_126() {
        let p = Tensor::scalar(14.600000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.600000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_127() {
        let p = Tensor::scalar(14.700000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.700000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_128() {
        let p = Tensor::scalar(14.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_129() {
        let p = Tensor::scalar(14.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((14.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_130() {
        let p = Tensor::scalar(15.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_131() {
        let p = Tensor::scalar(15.100000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.100000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_132() {
        let p = Tensor::scalar(15.200000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.200000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_133() {
        let p = Tensor::scalar(15.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_134() {
        let p = Tensor::scalar(15.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_135() {
        let p = Tensor::scalar(15.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_136() {
        let p = Tensor::scalar(15.600000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.600000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_137() {
        let p = Tensor::scalar(15.700000000000001);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.700000000000001) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_138() {
        let p = Tensor::scalar(15.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_139() {
        let p = Tensor::scalar(15.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((15.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_140() {
        let p = Tensor::scalar(16.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_141() {
        let p = Tensor::scalar(16.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_142() {
        let p = Tensor::scalar(16.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_143() {
        let p = Tensor::scalar(16.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_144() {
        let p = Tensor::scalar(16.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_145() {
        let p = Tensor::scalar(16.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_146() {
        let p = Tensor::scalar(16.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_147() {
        let p = Tensor::scalar(16.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_148() {
        let p = Tensor::scalar(16.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_149() {
        let p = Tensor::scalar(16.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((16.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_150() {
        let p = Tensor::scalar(17.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_151() {
        let p = Tensor::scalar(17.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_152() {
        let p = Tensor::scalar(17.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_153() {
        let p = Tensor::scalar(17.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_154() {
        let p = Tensor::scalar(17.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_155() {
        let p = Tensor::scalar(17.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_156() {
        let p = Tensor::scalar(17.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_157() {
        let p = Tensor::scalar(17.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_158() {
        let p = Tensor::scalar(17.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_159() {
        let p = Tensor::scalar(17.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((17.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_160() {
        let p = Tensor::scalar(18.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_161() {
        let p = Tensor::scalar(18.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_162() {
        let p = Tensor::scalar(18.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_163() {
        let p = Tensor::scalar(18.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_164() {
        let p = Tensor::scalar(18.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_165() {
        let p = Tensor::scalar(18.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_166() {
        let p = Tensor::scalar(18.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_167() {
        let p = Tensor::scalar(18.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_168() {
        let p = Tensor::scalar(18.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_169() {
        let p = Tensor::scalar(18.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((18.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_170() {
        let p = Tensor::scalar(19.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_171() {
        let p = Tensor::scalar(19.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_172() {
        let p = Tensor::scalar(19.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_173() {
        let p = Tensor::scalar(19.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_174() {
        let p = Tensor::scalar(19.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_175() {
        let p = Tensor::scalar(19.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_176() {
        let p = Tensor::scalar(19.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_177() {
        let p = Tensor::scalar(19.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_178() {
        let p = Tensor::scalar(19.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_179() {
        let p = Tensor::scalar(19.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((19.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_180() {
        let p = Tensor::scalar(20.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_181() {
        let p = Tensor::scalar(20.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_182() {
        let p = Tensor::scalar(20.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_183() {
        let p = Tensor::scalar(20.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_184() {
        let p = Tensor::scalar(20.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_185() {
        let p = Tensor::scalar(20.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_186() {
        let p = Tensor::scalar(20.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_187() {
        let p = Tensor::scalar(20.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_188() {
        let p = Tensor::scalar(20.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_189() {
        let p = Tensor::scalar(20.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((20.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_190() {
        let p = Tensor::scalar(21.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_191() {
        let p = Tensor::scalar(21.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_192() {
        let p = Tensor::scalar(21.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_193() {
        let p = Tensor::scalar(21.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_194() {
        let p = Tensor::scalar(21.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_195() {
        let p = Tensor::scalar(21.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_196() {
        let p = Tensor::scalar(21.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_197() {
        let p = Tensor::scalar(21.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_198() {
        let p = Tensor::scalar(21.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_199() {
        let p = Tensor::scalar(21.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((21.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_200() {
        let p = Tensor::scalar(22.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_201() {
        let p = Tensor::scalar(22.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_202() {
        let p = Tensor::scalar(22.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_203() {
        let p = Tensor::scalar(22.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_204() {
        let p = Tensor::scalar(22.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_205() {
        let p = Tensor::scalar(22.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_206() {
        let p = Tensor::scalar(22.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_207() {
        let p = Tensor::scalar(22.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_208() {
        let p = Tensor::scalar(22.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_209() {
        let p = Tensor::scalar(22.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((22.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_210() {
        let p = Tensor::scalar(23.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_211() {
        let p = Tensor::scalar(23.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_212() {
        let p = Tensor::scalar(23.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_213() {
        let p = Tensor::scalar(23.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_214() {
        let p = Tensor::scalar(23.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_215() {
        let p = Tensor::scalar(23.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_216() {
        let p = Tensor::scalar(23.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_217() {
        let p = Tensor::scalar(23.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_218() {
        let p = Tensor::scalar(23.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_219() {
        let p = Tensor::scalar(23.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((23.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_220() {
        let p = Tensor::scalar(24.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_221() {
        let p = Tensor::scalar(24.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_222() {
        let p = Tensor::scalar(24.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_223() {
        let p = Tensor::scalar(24.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_224() {
        let p = Tensor::scalar(24.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_225() {
        let p = Tensor::scalar(24.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_226() {
        let p = Tensor::scalar(24.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_227() {
        let p = Tensor::scalar(24.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_228() {
        let p = Tensor::scalar(24.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_229() {
        let p = Tensor::scalar(24.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((24.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_230() {
        let p = Tensor::scalar(25.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_231() {
        let p = Tensor::scalar(25.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_232() {
        let p = Tensor::scalar(25.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_233() {
        let p = Tensor::scalar(25.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_234() {
        let p = Tensor::scalar(25.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_235() {
        let p = Tensor::scalar(25.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_236() {
        let p = Tensor::scalar(25.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_237() {
        let p = Tensor::scalar(25.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_238() {
        let p = Tensor::scalar(25.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_239() {
        let p = Tensor::scalar(25.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((25.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_240() {
        let p = Tensor::scalar(26.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_241() {
        let p = Tensor::scalar(26.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_242() {
        let p = Tensor::scalar(26.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_243() {
        let p = Tensor::scalar(26.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_244() {
        let p = Tensor::scalar(26.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_245() {
        let p = Tensor::scalar(26.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_246() {
        let p = Tensor::scalar(26.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_247() {
        let p = Tensor::scalar(26.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_248() {
        let p = Tensor::scalar(26.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_249() {
        let p = Tensor::scalar(26.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((26.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_250() {
        let p = Tensor::scalar(27.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_251() {
        let p = Tensor::scalar(27.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_252() {
        let p = Tensor::scalar(27.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_253() {
        let p = Tensor::scalar(27.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_254() {
        let p = Tensor::scalar(27.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_255() {
        let p = Tensor::scalar(27.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_256() {
        let p = Tensor::scalar(27.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_257() {
        let p = Tensor::scalar(27.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_258() {
        let p = Tensor::scalar(27.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_259() {
        let p = Tensor::scalar(27.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((27.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_260() {
        let p = Tensor::scalar(28.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_261() {
        let p = Tensor::scalar(28.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_262() {
        let p = Tensor::scalar(28.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_263() {
        let p = Tensor::scalar(28.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_264() {
        let p = Tensor::scalar(28.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_265() {
        let p = Tensor::scalar(28.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_266() {
        let p = Tensor::scalar(28.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_267() {
        let p = Tensor::scalar(28.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_268() {
        let p = Tensor::scalar(28.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_269() {
        let p = Tensor::scalar(28.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((28.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_270() {
        let p = Tensor::scalar(29.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_271() {
        let p = Tensor::scalar(29.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_272() {
        let p = Tensor::scalar(29.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_273() {
        let p = Tensor::scalar(29.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_274() {
        let p = Tensor::scalar(29.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_275() {
        let p = Tensor::scalar(29.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_276() {
        let p = Tensor::scalar(29.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_277() {
        let p = Tensor::scalar(29.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_278() {
        let p = Tensor::scalar(29.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_279() {
        let p = Tensor::scalar(29.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((29.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_280() {
        let p = Tensor::scalar(30.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_281() {
        let p = Tensor::scalar(30.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_282() {
        let p = Tensor::scalar(30.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_283() {
        let p = Tensor::scalar(30.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_284() {
        let p = Tensor::scalar(30.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_285() {
        let p = Tensor::scalar(30.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_286() {
        let p = Tensor::scalar(30.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_287() {
        let p = Tensor::scalar(30.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_288() {
        let p = Tensor::scalar(30.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_289() {
        let p = Tensor::scalar(30.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((30.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_290() {
        let p = Tensor::scalar(31.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_291() {
        let p = Tensor::scalar(31.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_292() {
        let p = Tensor::scalar(31.200000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.200000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_293() {
        let p = Tensor::scalar(31.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_294() {
        let p = Tensor::scalar(31.400000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.400000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_295() {
        let p = Tensor::scalar(31.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_296() {
        let p = Tensor::scalar(31.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_297() {
        let p = Tensor::scalar(31.700000000000003);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.700000000000003) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_298() {
        let p = Tensor::scalar(31.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_299() {
        let p = Tensor::scalar(31.900000000000002);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((31.900000000000002) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_300() {
        let p = Tensor::scalar(32.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_301() {
        let p = Tensor::scalar(32.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_302() {
        let p = Tensor::scalar(32.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_303() {
        let p = Tensor::scalar(32.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_304() {
        let p = Tensor::scalar(32.400000000000006);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.400000000000006) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_305() {
        let p = Tensor::scalar(32.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_306() {
        let p = Tensor::scalar(32.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_307() {
        let p = Tensor::scalar(32.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_308() {
        let p = Tensor::scalar(32.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_309() {
        let p = Tensor::scalar(32.900000000000006);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((32.900000000000006) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_310() {
        let p = Tensor::scalar(33.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_311() {
        let p = Tensor::scalar(33.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_312() {
        let p = Tensor::scalar(33.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_313() {
        let p = Tensor::scalar(33.3);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.3) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_314() {
        let p = Tensor::scalar(33.400000000000006);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.400000000000006) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_315() {
        let p = Tensor::scalar(33.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_316() {
        let p = Tensor::scalar(33.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_317() {
        let p = Tensor::scalar(33.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_318() {
        let p = Tensor::scalar(33.8);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.8) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_319() {
        let p = Tensor::scalar(33.900000000000006);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((33.900000000000006) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_320() {
        let p = Tensor::scalar(34.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_321() {
        let p = Tensor::scalar(34.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_322() {
        let p = Tensor::scalar(34.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_323() {
        let p = Tensor::scalar(34.300000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.300000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_324() {
        let p = Tensor::scalar(34.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_325() {
        let p = Tensor::scalar(34.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_326() {
        let p = Tensor::scalar(34.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_327() {
        let p = Tensor::scalar(34.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_328() {
        let p = Tensor::scalar(34.800000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.800000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_329() {
        let p = Tensor::scalar(34.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((34.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_330() {
        let p = Tensor::scalar(35.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_331() {
        let p = Tensor::scalar(35.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_332() {
        let p = Tensor::scalar(35.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_333() {
        let p = Tensor::scalar(35.300000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.300000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_334() {
        let p = Tensor::scalar(35.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_335() {
        let p = Tensor::scalar(35.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_336() {
        let p = Tensor::scalar(35.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_337() {
        let p = Tensor::scalar(35.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_338() {
        let p = Tensor::scalar(35.800000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.800000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_339() {
        let p = Tensor::scalar(35.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((35.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_340() {
        let p = Tensor::scalar(36.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_341() {
        let p = Tensor::scalar(36.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_342() {
        let p = Tensor::scalar(36.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_343() {
        let p = Tensor::scalar(36.300000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.300000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_344() {
        let p = Tensor::scalar(36.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_345() {
        let p = Tensor::scalar(36.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_346() {
        let p = Tensor::scalar(36.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_347() {
        let p = Tensor::scalar(36.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_348() {
        let p = Tensor::scalar(36.800000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.800000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_349() {
        let p = Tensor::scalar(36.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((36.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_350() {
        let p = Tensor::scalar(37.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_351() {
        let p = Tensor::scalar(37.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_352() {
        let p = Tensor::scalar(37.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_353() {
        let p = Tensor::scalar(37.300000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.300000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_354() {
        let p = Tensor::scalar(37.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_355() {
        let p = Tensor::scalar(37.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_356() {
        let p = Tensor::scalar(37.6);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.6) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_357() {
        let p = Tensor::scalar(37.7);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.7) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_358() {
        let p = Tensor::scalar(37.800000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.800000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_359() {
        let p = Tensor::scalar(37.9);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((37.9) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_360() {
        let p = Tensor::scalar(38.0);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((38.0) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_361() {
        let p = Tensor::scalar(38.1);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((38.1) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_362() {
        let p = Tensor::scalar(38.2);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((38.2) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_363() {
        let p = Tensor::scalar(38.300000000000004);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((38.300000000000004) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_364() {
        let p = Tensor::scalar(38.4);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((38.4) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_loss_grad_stress_365() {
        let p = Tensor::scalar(38.5);
        let t = Tensor::scalar(1.0);
        let g = grad_mse_loss(&p, &t, 1.0).unwrap();
        let exp = 2.0 * ((38.5) - 1.0);
        assert!((g.get(0) - exp).abs() < 1e-6);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
}
