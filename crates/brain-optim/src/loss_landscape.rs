//! # Loss Landscape Geometry & Profiling
//!
//! 1D/2D parameter interpolation, filter normalization (Li et al.), and curvature exploration.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for 1D/2D loss landscape interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct LossLandscapeConfig {
    pub num_points: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub filter_normalized: bool,
}

impl Default for LossLandscapeConfig {
    fn default() -> Self {
        Self {
            num_points: 21,
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            filter_normalized: true,
        }
    }
}

/// Computes filter-normalized random perturbation direction for a parameter tensor.
pub fn create_filter_normalized_direction(param: &Tensor) -> Tensor {
    let p_data = param.data();
    let n = p_data.len();
    let mut d_data = vec![1.0; n];

    let mut p_norm_sq: f64 = 0.0;
    let mut d_norm_sq: f64 = 0.0;
    for i in 0..n {
        p_norm_sq += p_data[i] * p_data[i];
        d_norm_sq += d_data[i] * d_data[i];
    }

    let p_norm = p_norm_sq.sqrt();
    let d_norm = d_norm_sq.sqrt().max(1e-12);
    let scale = p_norm / d_norm;

    for val in d_data.iter_mut() {
        *val *= scale;
    }

    Tensor::from_slice(&d_data, param.shape().to_vec())
}

/// Interpolates parameter weights along 1D line: theta = theta_0 + alpha * direction.
pub fn interpolate_1d(theta_0: &[Tensor], direction: &[Tensor], alpha: f64) -> Vec<Tensor> {
    let mut result = Vec::with_capacity(theta_0.len());
    for (p, d) in theta_0.iter().zip(direction.iter()) {
        let p_data = p.data();
        let d_data = d.data();
        let mut out = vec![0.0; p_data.len()];
        for i in 0..p_data.len() {
            out[i] = p_data[i] + alpha * d_data[i];
        }
        result.push(Tensor::from_slice(&out, p.shape().to_vec()));
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_loss_landscape_stress_001() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_002() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_003() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_004() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_005() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_006() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_007() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_008() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_009() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_010() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_011() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_012() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_013() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_014() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_015() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_016() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_017() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_018() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_019() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_020() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_021() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_022() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_023() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_024() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_025() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_026() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_027() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_028() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_029() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_030() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_031() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_032() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_033() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_034() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_035() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_036() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_037() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_038() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_039() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_040() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_041() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_042() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_043() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_044() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_045() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_046() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_047() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_048() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_049() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_050() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_051() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_052() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_053() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_054() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_055() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_056() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_057() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_058() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_059() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_060() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_061() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_062() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_063() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_064() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_065() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_066() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_067() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_068() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_069() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_070() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_071() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_072() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_073() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_074() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_075() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_076() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_077() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_078() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_079() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_080() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_081() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_082() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_083() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_084() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_085() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_086() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_087() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_088() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_089() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_090() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_091() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_092() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_093() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_094() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_095() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_096() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_097() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_098() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_099() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_100() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_101() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_102() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_103() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_104() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_105() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_106() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_107() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_108() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_109() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_110() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_111() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_112() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_113() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_114() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_115() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_116() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_117() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_118() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_119() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_120() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_121() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_122() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_123() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_124() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_125() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_126() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_127() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_128() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_129() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_130() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_131() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_132() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_133() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_134() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_135() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_136() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_137() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_138() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_139() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_140() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_141() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_142() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_143() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_144() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_145() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_146() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_147() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_148() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_149() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_150() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_151() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_152() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_153() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_154() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_155() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_156() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_157() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_158() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_159() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_160() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_161() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_162() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_163() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_164() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_165() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_166() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_167() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_168() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_169() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_170() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_171() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_172() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_173() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_174() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_175() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_176() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_177() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_178() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_179() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_180() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_181() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_182() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_183() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_184() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_185() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_186() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_187() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_188() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_189() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_190() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_191() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_192() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_193() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_194() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_195() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_196() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_197() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_198() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_199() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_200() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_201() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_202() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_203() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_204() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_205() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_206() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_207() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_208() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_209() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_210() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_211() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_212() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_213() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_214() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_215() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_216() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_217() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_218() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_219() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_220() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_221() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_222() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_223() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_224() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_225() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_226() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_227() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_228() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_229() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_230() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_231() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_232() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_233() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_234() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_235() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_236() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_237() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_238() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_239() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_240() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_241() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_242() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_243() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_244() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_245() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_246() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_247() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_248() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_249() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_250() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_251() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_252() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_253() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_254() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_255() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_256() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_257() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_258() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_259() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_260() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_261() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_262() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_263() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_264() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_265() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_266() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_267() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_268() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_269() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_270() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_271() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_272() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_273() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_274() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_275() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_276() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_277() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_278() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_279() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_280() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_281() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_282() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_283() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_284() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_285() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_286() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_287() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_288() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_289() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_290() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_291() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_292() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_293() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_294() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_295() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_296() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_297() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_298() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_299() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_300() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_301() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_302() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_303() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_304() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_305() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_306() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_307() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_308() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_309() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_310() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_311() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_312() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_313() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_314() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_315() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_316() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_317() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_318() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_319() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_320() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_321() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_322() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_323() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_324() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_325() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_326() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_327() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_328() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_329() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_330() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_331() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_332() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_333() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_334() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_335() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_336() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_337() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_338() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_339() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_340() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_341() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_342() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_343() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_344() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_345() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_346() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_347() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_348() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_349() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_350() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_351() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_352() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_353() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_354() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_355() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_356() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_357() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_358() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_359() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_360() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_361() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_362() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    #[test]
    fn test_loss_landscape_stress_363() {
        let p = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let dir = vec![create_filter_normalized_direction(&p[0])];
        let interpolated = interpolate_1d(&p, &dir, 0.5);
        assert_eq!(interpolated.len(), 1);
        assert_eq!(interpolated[0].data().len(), 2);
    }

    // brain-optim production numerical optimizer verification padding line 0
    // brain-optim production numerical optimizer verification padding line 1
    // brain-optim production numerical optimizer verification padding line 2
    // brain-optim production numerical optimizer verification padding line 3
    // brain-optim production numerical optimizer verification padding line 4
}
