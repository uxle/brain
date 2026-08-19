//! # Auxiliary Quantization & Bit Packing Helpers
//!
//! Low-level byte and bit manipulations, tensor alignment, and clamping routines.
#![allow(missing_docs, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// Clamps a numeric slice in-place to bounds [min_v, max_v].
pub fn clamp_slice_in_place(data: &mut [f64], min_v: f64, max_v: f64) {
    for val in data.iter_mut() {
        *val = val.clamp(min_v, max_v);
    }
}

/// Packs 8-bit integers into packed 4-bit byte buffer.
pub fn pack_i8_to_i4_buffer(input: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((input.len() + 1) / 2);
    let mut i = 0;
    while i < input.len() {
        let low = input[i];
        let high = if i + 1 < input.len() { input[i + 1] } else { 0 };
        let packed = ((high as u8 & 0x0F) << 4) | (low as u8 & 0x0F);
        out.push(packed);
        i += 2;
    }
    out
}

/// Unpacks 4-bit byte buffer into 8-bit signed integer vector.
pub fn unpack_i4_to_i8_buffer(input: &[u8], original_len: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(original_len);
    for &byte in input {
        let mut low = (byte & 0x0F) as i8;
        if low >= 8 { low -= 16; }
        out.push(low);
        if out.len() == original_len { break; }

        let mut high = ((byte >> 4) & 0x0F) as i8;
        if high >= 8 { high -= 16; }
        out.push(high);
        if out.len() == original_len { break; }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
