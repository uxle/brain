//! # Message Framing & Transport Payloads
//!
//! Structured message headers, op tags, and fragmentation descriptors.

/// Frame header identifying the message source, destination, and payload size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageHeader {
    pub src_rank: usize,
    pub dest_rank: usize,
    pub tag: usize,
}

impl MessageHeader {
    /// Creates a new `MessageHeader`.
    pub fn new(src_rank: usize, dest_rank: usize, tag: usize) -> Self {
        Self {
            src_rank,
            dest_rank,
            tag,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
