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
}
