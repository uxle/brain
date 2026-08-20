//! # Genome Encoding Schemes
//!
//! Direct real-valued, binary bitstring, and discrete permutation encodings.
#![allow(missing_docs)]

/// Supported genome representation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodingKind {
    #[default]
    DirectReal,
    Binary,
    Permutation,
}

/// Encoding metadata and conversion helpers.
#[derive(Debug, Clone, Default)]
pub struct GenomeEncoding {
    pub kind: EncodingKind,
    pub dimension: usize,
}

impl GenomeEncoding {
    pub fn new(kind: EncodingKind, dimension: usize) -> Self {
        Self { kind, dimension }
    }

    pub fn is_valid_permutation(perm: &[usize]) -> bool {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &item in perm {
            if item >= n || seen[item] {
                return false;
            }
            seen[item] = true;
        }
        true
    }
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
