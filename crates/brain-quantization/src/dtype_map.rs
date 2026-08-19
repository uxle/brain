//! # Low-Bit Data Type Lowering & Casting
//!
//! Numeric conversion maps across floating point (f64/f32/bfloat16/float16) and integer (int8/uint8/int4) formats.
#![allow(missing_docs)]

/// Conversion mapping engine.
#[derive(Debug, Clone, Default)]
pub struct DTypeMap;

impl DTypeMap {
    /// Converts a 64-bit float to IEEE 754 half-precision float16 (represented as u16).
    pub fn f64_to_f16(val: f64) -> u16 {
        let f = val as f32;
        let bits = f.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;

        if exp == 0xFF {
            // Inf or NaN
            let f16_frac = if frac != 0 { 0x200 } else { 0 };
            return ((sign as u16) << 15) | 0x7C00 | f16_frac;
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            // Overflow to Inf
            return ((sign as u16) << 15) | 0x7C00;
        }
        if new_exp <= 0 {
            // Subnormal or underflow to 0
            return (sign as u16) << 15;
        }

        let new_frac = (frac >> 13) as u16;
        ((sign as u16) << 15) | ((new_exp as u16) << 10) | new_frac
    }

    /// Converts IEEE 754 half-precision float16 (u16) back to 64-bit float.
    pub fn f16_to_f64(bits: u16) -> f64 {
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1F;
        let frac = bits & 0x3FF;

        if exp == 0x1F {
            if frac != 0 {
                return f64::NAN;
            }
            return if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY };
        }
        if exp == 0 {
            if frac == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            // Subnormal
            let val = (frac as f64) / 1024.0 * 2.0f64.powi(-14);
            return if sign == 1 { -val } else { val };
        }

        let val = (1.0 + (frac as f64) / 1024.0) * 2.0f64.powi((exp as i32) - 15);
        if sign == 1 { -val } else { val }
    }

    /// Converts 64-bit float to Brain BFloat16 (top 16 bits of single precision float).
    pub fn f64_to_bf16(val: f64) -> u16 {
        let f = val as f32;
        let bits = f.to_bits();
        let lsb = (bits >> 15) & 1;
        let carry = (bits & 0x7FFF) + 0x7FFF + lsb;
        ((bits.wrapping_add(carry)) >> 16) as u16
    }

    /// Converts BFloat16 (u16) back to 64-bit float.
    pub fn bf16_to_f64(bits: u16) -> f64 {
        let f_bits = (bits as u32) << 16;
        f32::from_bits(f_bits) as f64
    }

    /// Packs two signed 4-bit integers into a single byte.
    pub fn pack_int4(low: i8, high: i8) -> u8 {
        let l = (low as u8) & 0x0F;
        let h = ((high as u8) & 0x0F) << 4;
        l | h
    }

    /// Unpacks a single byte into two signed 4-bit integers.
    pub fn unpack_int4(byte: u8) -> (i8, i8) {
        let mut low = (byte & 0x0F) as i8;
        if low >= 8 { low -= 16; }

        let mut high = ((byte >> 4) & 0x0F) as i8;
        if high >= 8 { high -= 16; }

        (low, high)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
