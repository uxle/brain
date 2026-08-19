//! # Quantization Mathematical Utilities
//!
//! Scale factor computation, zero-point derivation, percentile estimation, and entropy minimization.
#![allow(missing_docs)]

use super::core::{QuantDType, QuantError, QuantResult};

/// Finds the minimum and maximum floating point values across a slice.
pub fn minmax(data: &[f64]) -> QuantResult<(f64, f64)> {
    if data.is_empty() {
        return Err(QuantError::EmptyTensor);
    }
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in data {
        if !v.is_nan() && !v.is_infinite() {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
    }
    if min_val.is_infinite() || max_val.is_infinite() {
        min_val = 0.0;
        max_val = 0.0;
    }
    Ok((min_val, max_val))
}

/// Derives affine scale factor and zero-point from observed min/max bounds.
pub fn compute_scale_zero_point(min_val: f64, max_val: f64, dtype: QuantDType, symmetric: bool) -> QuantResult<(f64, i32)> {
    let qmin = dtype.qmin() as f64;
    let qmax = dtype.qmax() as f64;

    if symmetric {
        let max_abs = min_val.abs().max(max_val.abs()).max(1e-8);
        let scale = (2.0 * max_abs) / (qmax - qmin);
        let zero_point = 0;
        Ok((scale.max(1e-12), zero_point))
    } else {
        let range = (max_val - min_val).max(1e-8);
        let scale = range / (qmax - qmin);
        let zero_point = (qmin - min_val / scale).round() as i32;
        let clamped_zp = zero_point.clamp(dtype.qmin(), dtype.qmax());
        Ok((scale.max(1e-12), clamped_zp))
    }
}

/// Computes the requested percentile value from a numeric slice.
pub fn percentile_slice(data: &[f64], p: f64) -> QuantResult<f64> {
    if data.is_empty() {
        return Err(QuantError::EmptyTensor);
    }
    let mut valid_data: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan() && !v.is_infinite()).collect();
    if valid_data.is_empty() {
        return Ok(0.0);
    }
    valid_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((valid_data.len() - 1) as f64 * (p.clamp(0.0, 100.0) / 100.0)).round() as usize;
    Ok(valid_data[idx])
}

/// Simulates affine integer quantization on floating point values.
pub fn quantize_val(val: f64, scale: f64, zero_point: i32, qmin: i32, qmax: i32) -> i32 {
    let q = (val / scale).round() as i32 + zero_point;
    q.clamp(qmin, qmax)
}

/// Dequantizes an integer value back to floating point representation.
pub fn dequantize_val(q: i32, scale: f64, zero_point: i32) -> f64 {
    (q - zero_point) as f64 * scale
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
