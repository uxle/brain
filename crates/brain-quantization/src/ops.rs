//! # Quantization Integer Math Operations
//!
//! Low-level kernels for requantization, per-channel scaling, and saturating arithmetic.
#![allow(missing_docs)]

use super::core::{QuantError, QuantResult};

/// Requantizes 32-bit accumulators into 8-bit output integers given multiplier scale and zero-point.
pub fn requantize(
    accumulators: &[i32],
    effective_scale: f64,
    output_zp: i32,
    qmin: i32,
    qmax: i32,
) -> Vec<i32> {
    let mut out = Vec::with_capacity(accumulators.len());
    for &val in accumulators {
        let real = val as f64 * effective_scale;
        let q = real.round() as i32 + output_zp;
        out.push(q.clamp(qmin, qmax));
    }
    out
}

/// Applies per-channel scale factors to 32-bit integer matrix columns.
pub fn requantize_per_channel(
    accumulators: &[i32],
    scales: &[f64],
    output_zp: i32,
    m: usize,
    n: usize,
    qmin: i32,
    qmax: i32,
) -> QuantResult<Vec<i32>> {
    if scales.len() != n || accumulators.len() != m * n {
        return Err(QuantError::ChannelCountMismatch {
            expected: n,
            found: scales.len(),
        });
    }

    let mut out = Vec::with_capacity(m * n);
    for row in 0..m {
        for col in 0..n {
            let val = accumulators[row * n + col];
            let real = val as f64 * scales[col];
            let q = real.round() as i32 + output_zp;
            out.push(q.clamp(qmin, qmax));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
