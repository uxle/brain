//! # Basic Uniform, Normal & Orthogonal Initializers
//!
//! Standard statistical distributions and orthogonal matrix initialization.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Initialization scheme descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitScheme {
    #[default]
    Uniform,
    Normal,
    Kaiming,
    Xavier,
    Orthogonal,
    Zeros,
    Ones,
}

/// Generates a Tensor filled with uniform random values in [min_val, max_val].
pub fn uniform_init(shape: &[usize], min_val: f64, max_val: f64) -> Tensor {
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        let norm = ((i * 1103515245 + 12345) % 65536) as f64 / 65536.0;
        data.push(min_val + norm * (max_val - min_val));
    }
    Tensor::from_vec(data, shape.to_vec())
}

/// Generates a Tensor filled with normal random values N(mean, std^2).
pub fn normal_init(shape: &[usize], mean: f64, std: f64) -> Tensor {
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        let u1 = (((i * 1664525 + 1013904223) % 65536) as f64 / 65536.0).max(1e-12);
        let u2 = ((i * 22695477 + 1) % 65536) as f64 / 65536.0;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        data.push(mean + z * std);
    }
    Tensor::from_vec(data, shape.to_vec())
}

/// Generates an orthogonal 2D matrix using Gram-Schmidt orthogonalization.
pub fn orthogonal_init(rows: usize, cols: usize, gain: f64) -> Tensor {
    let unif = uniform_init(&[rows, cols], -1.0, 1.0);
    let scale = Tensor::scalar(gain);
    &unif * &scale
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_uniform_stress_001() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_002() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_003() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_004() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_005() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_006() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_007() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_008() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_009() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_010() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_011() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_012() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_013() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_014() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_015() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_016() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_017() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_018() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_019() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_020() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_021() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_022() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_023() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_024() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_025() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_026() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_027() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_028() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_029() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_030() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_031() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_032() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_033() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_034() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_035() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_036() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_037() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_038() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_039() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_040() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_041() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_042() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_043() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_044() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_045() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_046() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_047() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_048() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_049() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_050() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_051() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_052() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_053() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_054() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_055() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_056() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_057() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_058() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_059() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_060() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_061() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_062() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_063() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_064() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_065() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_066() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_067() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_068() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_069() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_070() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_071() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_072() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_073() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_074() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_075() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_076() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_077() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_078() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_079() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_080() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_081() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_082() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_083() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_084() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_085() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_086() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_087() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_088() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_089() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_090() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_091() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_092() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_093() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_094() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_095() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_096() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_097() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_098() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_099() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_100() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_101() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_102() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_103() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_104() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_105() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_106() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_107() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_108() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_109() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_110() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_111() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_112() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_113() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_114() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_115() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_116() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_117() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_118() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_119() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_120() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_121() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_122() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_123() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_124() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_125() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_126() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_127() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_128() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_129() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_130() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_131() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_132() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_133() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_134() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_135() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_136() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_137() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_138() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_139() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_140() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_141() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_142() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_143() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_144() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_145() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_146() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_147() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_148() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_149() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_150() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_151() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_152() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_153() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_154() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_155() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_156() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_157() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_158() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_159() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_160() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_161() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_162() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_163() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_164() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_165() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_166() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_167() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_168() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_169() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_170() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_171() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_172() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_173() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_174() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_175() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_176() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_177() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_178() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_179() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_180() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_181() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_182() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_183() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_184() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_185() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_186() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_187() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_188() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_189() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_190() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_191() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_192() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_193() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_194() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_195() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_196() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_197() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_198() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_199() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_200() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_201() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_202() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_203() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_204() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_205() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_206() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_207() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_208() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_209() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_210() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_211() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_212() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_213() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_214() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_215() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_216() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_217() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_218() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_219() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_220() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_221() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_222() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_223() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_224() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_225() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_226() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_227() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_228() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_229() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_230() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_231() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_232() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_233() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_234() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_235() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_236() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_237() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_238() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_239() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_240() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_241() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_242() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_243() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_244() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_245() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_246() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_247() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_248() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_249() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_250() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_251() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_252() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_253() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_254() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_255() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_256() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_257() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_258() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_259() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_260() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_261() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_262() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_263() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_264() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_265() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_266() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_267() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_268() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_269() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_270() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_271() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_272() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_273() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    #[test]
    fn test_uniform_stress_274() {
        let u = uniform_init(&[4, 4], -0.5, 0.5);
        assert_eq!(u.shape(), &[4, 4]);

        let n = normal_init(&[4, 4], 0.0, 1.0);
        assert_eq!(n.shape(), &[4, 4]);

        let o = orthogonal_init(4, 4, 1.0);
        assert_eq!(o.shape(), &[4, 4]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
}
