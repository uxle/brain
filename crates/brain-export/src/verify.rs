//! # Numerical Round-Trip Output Verification
//!
//! Compares reference outputs against exported models to ensure mathematical equivalence.

use crate::core::ExportError;
use brain_core::Tensor;

/// Verifies that model outputs match expected reference tensors within tolerance.
pub fn verify_export(actual: &Tensor, expected: &Tensor, tol: f64) -> Result<(), ExportError> {
    let diff = actual - expected;
    let max_err = diff.to_vec().iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    if max_err > tol {
        return Err(ExportError::VerificationFailed(format!(
            "Max error {} exceeded tolerance {}",
            max_err, tol
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
