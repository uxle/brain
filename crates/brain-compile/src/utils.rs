//! # Compiler Utilities, Hashing & Formatting
//!
//! Provides FNV-1a graph hashing, topological sorting, and FLOP/byte formatting helpers.

/// Computes a 64-bit FNV-1a hash of a byte slice.
pub fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Formats a byte count into a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Formats a FLOP count into a human-readable string.
pub fn format_flops(flops: u64) -> String {
    if flops < 1_000 {
        format!("{} FLOP", flops)
    } else if flops < 1_000_000 {
        format!("{:.2} KFLOP", flops as f64 / 1_000.0)
    } else if flops < 1_000_000_000 {
        format!("{:.2} MFLOP", flops as f64 / 1_000_000.0)
    } else {
        format!("{:.2} GFLOP", flops as f64 / 1_000_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
