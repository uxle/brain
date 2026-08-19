//! # Computer Vision Operations & Geometry Helpers
//!
//! Provides bounding box mathematics, affine grids, grid sampling, and histogram equalization.

pub mod boxes;
pub mod geometry;
pub mod hist_eq;

pub use boxes::{box_area, box_iou_matrix};
pub use geometry::{affine_grid, grid_sample};
pub use hist_eq::{equalize_histogram, ColorSpace};

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
