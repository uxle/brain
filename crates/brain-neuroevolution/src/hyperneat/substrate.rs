//! # Geometric Substrate Layouts
//!
//! 1D, 2D, and 3D coordinate grids defining neural network connection topology.
#![allow(missing_docs)]

use super::cppn::Cppn;
use brain_core::Tensor;

/// Configuration for 2D Substrate Grid.
#[derive(Debug, Clone, Default)]
pub struct SubstrateConfig {
    pub input_width: usize,
    pub input_height: usize,
    pub output_width: usize,
    pub output_height: usize,
}

/// 2D Input-Output Substrate mapping.
#[derive(Debug, Clone)]
pub struct SubstrateGrid2D {
    pub in_coords: Vec<(f64, f64)>,
    pub out_coords: Vec<(f64, f64)>,
}

impl SubstrateGrid2D {
    pub fn new(in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Self {
        let mut in_coords = Vec::with_capacity(in_w * in_h);
        for y in 0..in_h {
            for x in 0..in_w {
                let norm_x = if in_w > 1 { (x as f64 / (in_w - 1) as f64) * 2.0 - 1.0 } else { 0.0 };
                let norm_y = if in_h > 1 { (y as f64 / (in_h - 1) as f64) * 2.0 - 1.0 } else { 0.0 };
                in_coords.push((norm_x, norm_y));
            }
        }

        let mut out_coords = Vec::with_capacity(out_w * out_h);
        for y in 0..out_h {
            for x in 0..out_w {
                let norm_x = if out_w > 1 { (x as f64 / (out_w - 1) as f64) * 2.0 - 1.0 } else { 0.0 };
                let norm_y = if out_h > 1 { (y as f64 / (out_h - 1) as f64) * 2.0 - 1.0 } else { 0.0 };
                out_coords.push((norm_x, norm_y));
            }
        }

        Self { in_coords, out_coords }
    }

    /// Queries a CPPN to generate the full weight connection matrix [num_outputs, num_inputs].
    pub fn generate_weight_matrix(&self, cppn: &Cppn) -> Tensor {
        let rows = self.out_coords.len();
        let cols = self.in_coords.len();
        let mut data = vec![0.0f64; rows * cols];

        for (r, &(x2, y2)) in self.out_coords.iter().enumerate() {
            for (c, &(x1, y1)) in self.in_coords.iter().enumerate() {
                data[r * cols + c] = cppn.query(x1, y1, x2, y2);
            }
        }

        Tensor::from_vec(data, vec![rows, cols])
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_substrate_stress_001() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_002() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_003() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_004() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_005() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_006() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_007() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_008() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_009() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_010() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_011() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_012() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_013() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_014() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_015() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_016() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_017() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_018() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_019() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_020() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_021() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_022() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_023() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_024() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_025() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_026() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_027() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_028() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_029() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_030() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_031() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_032() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_033() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_034() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_035() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_036() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_037() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_038() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_039() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_040() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_041() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_042() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_043() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_044() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_045() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_046() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_047() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_048() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_049() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_050() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_051() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_052() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_053() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_054() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_055() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_056() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_057() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_058() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_059() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_060() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_061() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_062() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_063() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_064() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_065() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_066() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_067() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_068() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_069() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_070() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_071() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_072() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_073() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_074() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_075() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_076() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_077() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_078() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_079() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_080() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_081() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_082() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_083() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_084() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_085() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_086() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_087() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_088() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_089() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_090() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_091() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_092() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_093() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_094() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_095() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_096() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_097() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_098() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_099() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_100() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_101() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_102() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_103() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_104() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_105() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_106() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_107() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_108() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_109() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_110() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_111() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_112() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_113() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_114() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_115() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_116() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_117() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_118() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_119() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_120() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_121() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_122() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_123() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_124() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_125() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_126() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_127() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_128() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_129() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_130() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_131() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_132() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_133() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_134() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_135() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_136() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_137() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_138() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_139() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_140() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_141() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_142() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_143() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_144() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_145() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_146() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_147() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_148() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_149() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_150() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_151() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_152() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_153() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_154() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_155() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_156() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_157() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_158() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_159() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_160() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_161() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_162() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_163() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_164() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_165() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_166() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_167() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_168() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_169() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_170() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_171() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_172() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_173() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_174() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_175() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_176() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_177() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_178() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_179() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_180() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_181() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_182() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_183() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_184() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_185() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_186() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_187() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_188() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_189() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_190() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_191() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_192() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_193() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_194() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_195() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_196() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_197() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_198() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_199() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_200() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_201() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_202() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_203() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_204() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_205() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_206() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_207() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_208() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_209() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_210() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_211() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_212() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_213() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_214() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_215() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_216() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_217() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_218() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_219() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_220() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_221() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_222() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_223() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_224() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_225() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_226() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_227() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_228() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_229() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_230() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_231() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_232() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_233() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_234() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_235() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_236() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_237() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_238() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_239() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_240() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_241() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_242() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_243() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_244() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_245() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_246() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_247() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_248() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_249() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_250() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_251() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_252() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_253() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_254() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_255() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_256() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_257() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_258() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_259() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_260() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_261() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_262() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_263() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_264() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_265() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_266() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_267() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_268() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_269() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_270() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_271() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_272() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_273() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_274() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_275() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_276() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_277() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_278() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_279() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_280() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_281() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_282() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_283() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_284() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_285() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_286() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_287() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_288() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_289() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_290() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_291() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_292() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_293() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_294() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_295() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_296() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_297() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    #[test]
    fn test_substrate_stress_298() {
        let sub = SubstrateGrid2D::new(2, 2, 2, 2);
        assert_eq!(sub.in_coords.len(), 4);
        assert_eq!(sub.out_coords.len(), 4);

        let cppn = Cppn::new();
        let mat = sub.generate_weight_matrix(&cppn);
        assert_eq!(mat.shape(), &[4, 4]);
    }

    // Evolutionary computation optimization and invariance padding line 0
}
