//! # Pure-Rust Numeric Compression Codecs
//!
//! Run-Length Encoding (RLE) and Delta encoding for compression of numeric data arrays.

/// Encodes a slice of integers using Run-Length Encoding.
pub fn rle_encode(data: &[i64]) -> Vec<(i64, usize)> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur = data[0];
    let mut count = 1;

    for &val in &data[1..] {
        if val == cur {
            count += 1;
        } else {
            out.push((cur, count));
            cur = val;
            count = 1;
        }
    }
    out.push((cur, count));
    out
}

/// Applies delta encoding to numeric sequence.
pub fn delta_encode(data: &[i64]) -> Vec<i64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len());
    out.push(data[0]);
    for i in 1..data.len() {
        out.push(data[i] - data[i - 1]);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
