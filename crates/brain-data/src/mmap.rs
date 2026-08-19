//! # Zero-Copy Binary Chunk Reader
//!
//! Buffered binary chunk ingestion with zero-copy slicing abstractions.

/// Fast binary chunk reader.
pub struct MmapChunkReader {
    buffer: Vec<u8>,
}

impl MmapChunkReader {
    /// Creates a reader wrapping raw bytes.
    pub fn from_bytes(buffer: Vec<u8>) -> Self {
        Self { buffer }
    }

    /// Reads a slice from the buffer.
    pub fn read_slice(&self, start: usize, len: usize) -> Option<&[u8]> {
        if start + len <= self.buffer.len() {
            Some(&self.buffer[start..start + len])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
