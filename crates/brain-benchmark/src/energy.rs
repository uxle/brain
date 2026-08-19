//! # Software Energy and Power Estimation
//!
//! Models energy consumption in Joules and compute efficiency in GFLOPS/Watt.

use std::time::Duration;

/// Models compute energy and power efficiency.
pub struct EnergyEstimator;

impl EnergyEstimator {
    /// Estimates energy consumption in Joules given duration and estimated wattage.
    pub fn estimate_joules(duration: Duration, estimated_watts: f64) -> f64 {
        duration.as_secs_f64() * estimated_watts
    }

    /// Estimates compute efficiency in GigaFLOPS per Watt.
    pub fn compute_efficiency_gflops_per_watt(gflops: f64, estimated_watts: f64) -> f64 {
        if estimated_watts <= 0.0 {
            0.0
        } else {
            gflops / estimated_watts
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
