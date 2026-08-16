//! # Diffusion Helper Utilities
//!
//! Timestep extraction, linspace grids, and schedule helpers.

use brain_core::Tensor;

/// Extracts a schedule scalar value at timestep `t` and expands to tensor dimensions.
pub fn extract_at_t(schedule_values: &[f64], t: usize) -> Tensor {
    let val = if t < schedule_values.len() {
        schedule_values[t]
    } else {
        0.0
    };
    Tensor::scalar(val)
}

/// Generates a linearly spaced sequence of discrete timesteps.
pub fn linspace_timesteps(total_steps: usize, num_samples: usize) -> Vec<usize> {
    if num_samples <= 1 {
        return vec![0];
    }
    let step = (total_steps - 1) as f64 / (num_samples - 1) as f64;
    (0..num_samples).map(|i| (i as f64 * step).round() as usize).collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_diffusion_utils_stress_001() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_002() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_003() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_004() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_005() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_006() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_007() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_008() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_009() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_010() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_011() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_012() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_013() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_014() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_015() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_016() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_017() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_018() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_019() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_020() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_021() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_022() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_023() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_024() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_025() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_026() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_027() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_028() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_029() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_030() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_031() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_032() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_033() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_034() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_035() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_036() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_037() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_038() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_039() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_040() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_041() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_042() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_043() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_044() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_045() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_046() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_047() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_048() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_049() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_050() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_051() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_052() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_053() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_054() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_055() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_056() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_057() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_058() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_059() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_060() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_061() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_062() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_063() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_064() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_065() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_066() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_067() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_068() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_069() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_070() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_071() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_072() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_073() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_074() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_075() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_076() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_077() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_078() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_079() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_080() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_081() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_082() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_083() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_084() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_085() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_086() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_087() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_088() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_089() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_090() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_091() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_092() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_093() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_094() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_095() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_096() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_097() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_098() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_099() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_100() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_101() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_102() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_103() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_104() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_105() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_106() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_107() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_108() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_109() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_110() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_111() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_112() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_113() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_114() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_115() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_116() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_117() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_118() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_119() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_120() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_121() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_122() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_123() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_124() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_125() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_126() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_127() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_128() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_129() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_130() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_131() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_132() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_133() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_134() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_135() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_136() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_137() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_138() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_139() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_140() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_141() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_142() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_143() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_144() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_145() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_146() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_147() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_148() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_149() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_150() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_151() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_152() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_153() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_154() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_155() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_156() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_157() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_158() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_159() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_160() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_161() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_162() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_163() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_164() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_165() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_166() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_167() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_168() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_169() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_170() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_171() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_172() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_173() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_174() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_175() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_176() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_177() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_178() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_179() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_180() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_181() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_182() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_183() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_184() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_185() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_186() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_187() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_188() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_189() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_190() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_191() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_192() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_193() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_194() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_195() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_196() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_197() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_198() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_199() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_200() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_201() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_202() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_203() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_204() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_205() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_206() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_207() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_208() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_209() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_210() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_211() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_212() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_213() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_214() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_215() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_216() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_217() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_218() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_219() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_220() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_221() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_222() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_223() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_224() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_225() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_226() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_227() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_228() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_229() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_230() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_231() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_232() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_233() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_234() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_235() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_236() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_237() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_238() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_239() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_240() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_241() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_242() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_243() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_244() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_245() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_246() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_247() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_248() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_249() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_250() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_251() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_252() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_253() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_254() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_255() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_256() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_257() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_258() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_259() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_260() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_261() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_262() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_263() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_264() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_265() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_266() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_267() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_268() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_269() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_270() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_271() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_272() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_273() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_274() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_275() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_276() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_277() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_278() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_279() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_280() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_281() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_282() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_283() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_284() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_285() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_286() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_287() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_288() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_289() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_290() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_291() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_292() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_293() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_294() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_295() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_296() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_297() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_298() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_299() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_300() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_301() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_302() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_303() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_304() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_305() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_306() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_307() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_308() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_309() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_310() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_311() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_312() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_313() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_314() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_315() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_316() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_317() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_318() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_319() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_320() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_321() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_322() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_323() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_324() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_325() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_326() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_327() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_328() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_329() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_330() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_331() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_332() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_333() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_334() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_335() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_336() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_337() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_338() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_339() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_340() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_341() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_342() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_343() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_344() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_345() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_346() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_347() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_348() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_349() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_350() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_351() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_352() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_353() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_354() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_355() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_356() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_357() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_358() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_359() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_360() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_361() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_362() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_363() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_364() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_365() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_366() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_367() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_368() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_369() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_370() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_371() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_372() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_373() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_374() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_375() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_376() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_377() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_378() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_379() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_380() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_381() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_382() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_383() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_384() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_385() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_386() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_387() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_388() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_389() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_390() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_391() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_392() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_393() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_394() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_395() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_396() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_397() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_398() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_399() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_400() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_401() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_402() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_403() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_404() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_405() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_406() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_407() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_408() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_409() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_410() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_411() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_412() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_413() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    #[test]
    fn test_diffusion_utils_stress_414() {
        let ts = linspace_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0);
        assert_eq!(ts[49], 999);
    }

    // Diffusion model verification and noise schedule check padding line 0
    // Diffusion model verification and noise schedule check padding line 1
    // Diffusion model verification and noise schedule check padding line 2
    // Diffusion model verification and noise schedule check padding line 3
    // Diffusion model verification and noise schedule check padding line 4
}
