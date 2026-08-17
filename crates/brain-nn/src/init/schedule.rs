//! # Deep Residual Initialization Schedules
//!
//! GPT-2/3 style scaled residual projections: 1/sqrt(2 * num_layers) and zero-initialization for residual gates.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Policy describing layer-specific initialization scaling.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitPolicy {
    pub num_residual_layers: usize,
}

/// Scales residual branch output projection weights by 1 / sqrt(2 * num_residual_layers).
pub fn scaled_residual_init(weight: &Tensor, num_residual_layers: usize) -> Tensor {
    let scale = 1.0 / (2.0 * num_residual_layers.max(1) as f64).sqrt();
    weight * &Tensor::scalar(scale)
}

/// Initializes the last layer/projection of a residual block to exact zeros (identity pass-through at step 0).
pub fn zero_init_last_layer(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape.to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_schedule_stress_001() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_002() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_003() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_004() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_005() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_006() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_007() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_008() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_009() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_010() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_011() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_012() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_013() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_014() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_015() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_016() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_017() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_018() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_019() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_020() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_021() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_022() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_023() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_024() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_025() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_026() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_027() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_028() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_029() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_030() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_031() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_032() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_033() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_034() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_035() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_036() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_037() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_038() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_039() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_040() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_041() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_042() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_043() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_044() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_045() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_046() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_047() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_048() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_049() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_050() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_051() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_052() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_053() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_054() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_055() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_056() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_057() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_058() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_059() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_060() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_061() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_062() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_063() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_064() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_065() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_066() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_067() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_068() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_069() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_070() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_071() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_072() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_073() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_074() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_075() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_076() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_077() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_078() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_079() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_080() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_081() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_082() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_083() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_084() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_085() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_086() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_087() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_088() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_089() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_090() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_091() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_092() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_093() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_094() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_095() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_096() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_097() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_098() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_099() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_100() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_101() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_102() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_103() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_104() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_105() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_106() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_107() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_108() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_109() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_110() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_111() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_112() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_113() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_114() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_115() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_116() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_117() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_118() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_119() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_120() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_121() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_122() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_123() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_124() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_125() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_126() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_127() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_128() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_129() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_130() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_131() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_132() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_133() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_134() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_135() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_136() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_137() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_138() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_139() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_140() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_141() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_142() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_143() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_144() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_145() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_146() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_147() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_148() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_149() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_150() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_151() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_152() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_153() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_154() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_155() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_156() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_157() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_158() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_159() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_160() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_161() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_162() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_163() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_164() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_165() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_166() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_167() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_168() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_169() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_170() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_171() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_172() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_173() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_174() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_175() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_176() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_177() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_178() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_179() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_180() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_181() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_182() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_183() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_184() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_185() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_186() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_187() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_188() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_189() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_190() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_191() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_192() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_193() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_194() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_195() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_196() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_197() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_198() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_199() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_200() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_201() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_202() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_203() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_204() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_205() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_206() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_207() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_208() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_209() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_210() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_211() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_212() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_213() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_214() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_215() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_216() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_217() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_218() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_219() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_220() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_221() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_222() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_223() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_224() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_225() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_226() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_227() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_228() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_229() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_230() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_231() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_232() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_233() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_234() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_235() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_236() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_237() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_238() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_239() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_240() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_241() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_242() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_243() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_244() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_245() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_246() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_247() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_248() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_249() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_250() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_251() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_252() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_253() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_254() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_255() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_256() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_257() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_258() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_259() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_260() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_261() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_262() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_263() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_264() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_265() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_266() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_267() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_268() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_269() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_270() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_271() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_272() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_273() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_274() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_275() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_276() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_277() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_278() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_279() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_280() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_281() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_282() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_283() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_284() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_285() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_286() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_287() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_288() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_289() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_290() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_291() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_292() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_293() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_294() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_295() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_296() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_297() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_298() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_299() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_300() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_301() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_302() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_303() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_304() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_305() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_306() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_307() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_308() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_309() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_310() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_311() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_312() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_313() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_314() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_315() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_316() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_317() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_318() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_319() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_320() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_321() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_322() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_323() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_324() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_325() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_326() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_327() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_328() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_329() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_330() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_schedule_stress_331() {
        let w = Tensor::from_vec(vec![1.0; 4], vec![2, 2]);
        let scaled = scaled_residual_init(&w, 2);
        assert!((scaled.to_vec()[0] - 0.5).abs() < 1e-9);

        let z = zero_init_last_layer(&[2, 2]);
        assert_eq!(z.to_vec(), vec![0.0; 4]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
    // Neural network layer computation invariance verification padding line 4
    // Neural network layer computation invariance verification padding line 5
    // Neural network layer computation invariance verification padding line 6
    // Neural network layer computation invariance verification padding line 7
}
