//! # Pipeline Helper Utilities & Hashing
//!
//! Provides FNV-1a hashing, iterator interleaving, and deduplication helpers.

/// Computes 64-bit FNV-1a hash of a byte slice.
pub fn fnv_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Deduplicates a slice of items while preserving order.
pub fn dedup_items<T: PartialEq + Clone>(items: &[T]) -> Vec<T> {
    let mut out = Vec::new();
    for it in items {
        if !out.contains(it) {
            out.push(it.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
