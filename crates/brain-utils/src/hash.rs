//! # Hashing Algorithms
//!
//! Provides fast, non-cryptographic hashing algorithms including FNV-1a (32/64-bit),
//! Murmur3-lite, xxHash-lite, and hash combination combinators.

/// FNV-1a 64-bit non-cryptographic hash of raw byte slice.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// FNV-1a 32-bit hash.
pub fn fnv1a_32(bytes: &[u8]) -> u32 {
    const FNV_OFFSET: u32 = 0x811c9dc5;
    const FNV_PRIME: u32 = 0x01000193;
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Combines two 64-bit hash values using Boost-style mix.
pub fn hash_combine_64(h1: u64, h2: u64) -> u64 {
    h1 ^ (h2.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(h1 << 6).wrapping_add(h1 >> 2))
}

/// Murmur3 32-bit hash implementation.
pub fn murmur3_32(bytes: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;
    let mut h = seed;
    let mut chunks = bytes.chunks_exact(4);
    for chunk in chunks.by_ref() {
        let mut k = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        k = k.wrapping_mul(C1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13);
        h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
    }
    let rem = chunks.remainder();
    let mut k = 0u32;
    for (i, &b) in rem.iter().enumerate() {
        k |= (b as u32) << (i * 8);
    }
    if !rem.is_empty() {
        k = k.wrapping_mul(C1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(C2);
        h ^= k;
    }
    h ^= bytes.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// Convenience string hasher using 64-bit FNV-1a.
pub fn hash_str(s: &str) -> u64 {
    fnv1a_64(s.as_bytes())
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_hashing_algorithms_1() {
        let data = b"brain deep learning framework";
        let h64 = fnv1a_64(data);
        let h32 = fnv1a_32(data);
        assert_ne!(h64, 0);
        assert_ne!(h32, 0);
    
        let m3 = murmur3_32(data, 0);
        assert_ne!(m3, 0);
    
        let combined = hash_combine_64(h64, m3 as u64);
        assert_ne!(combined, h64);
        assert_ne!(combined, m3 as u64);
    
        assert_eq!(hash_str("test_string"), fnv1a_64(b"test_string"));
    }
}
