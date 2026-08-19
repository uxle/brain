//! # Pattern Matching & Broadcasting Shape Helpers
//!
//! Utilities for subgraph pattern matching and tensor shape compatibility checks.

/// Computes the broadcasted output shape from two input shapes.
pub fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut out = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if dim_a == dim_b {
            out.push(dim_a);
        } else if dim_a == 1 {
            out.push(dim_b);
        } else if dim_b == 1 {
            out.push(dim_a);
        } else {
            return None;
        }
    }

    out.reverse();
    Some(out)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
