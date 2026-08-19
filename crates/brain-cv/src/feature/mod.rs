//! # Deep Feature Extraction Subsystem
//!
//! Multi-scale feature extraction backbones and Feature Pyramid Networks (FPN).

pub mod backbones;
pub mod fpn;

pub use backbones::BackboneZoo;
pub use fpn::Fpn;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
