//! # Weight Initialization Schemes
//!
//! Kaiming/He, Xavier/Glorot, Orthogonal, Normal, Uniform, and residual network initialization schedules.
#![allow(missing_docs)]

pub mod kaiming;
pub mod uniform;
pub mod schedule;

pub use kaiming::{kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal, InitConfig};
pub use uniform::{uniform_init, normal_init, orthogonal_init, InitScheme};
pub use schedule::{scaled_residual_init, zero_init_last_layer, InitPolicy};


/// Fan-in and Fan-out calculation from weight shape.
pub fn calculate_fan(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() { return (1, 1); }
    if shape.len() == 1 { return (shape[0], shape[0]); }
    if shape.len() == 2 { return (shape[1], shape[0]); } // [out_features, in_features]

    // Conv weights: [out_channels, in_channels, k_h, k_w, ...]
    let receptive_field: usize = shape[2..].iter().product();
    let fan_in = shape[1] * receptive_field;
    let fan_out = shape[0] * receptive_field;
    (fan_in, fan_out)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
}
