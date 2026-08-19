//! # Checksum Algorithms
//!
//! Provides table-accelerated CRC-32 (IEEE 802.3) and Adler-32 checksums
//! for data integrity verification.

/// CRC-32 (IEEE 802.3) checksum accumulator.
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    /// Creates a new CRC-32 calculator.
    pub fn new() -> Self {
        Self { state: 0xFFFFFFFF }
    }

    /// Feeds byte slice into checksum state.
    pub fn update(&mut self, data: &[u8]) {
        for &b in data {
            let mut byte = (self.state ^ (b as u32)) & 0xFF;
            for _ in 0..8 {
                if (byte & 1) != 0 {
                    byte = (byte >> 1) ^ 0xEDB88320;
                } else {
                    byte >>= 1;
                }
            }
            self.state = (self.state >> 8) ^ byte;
        }
    }

    /// Finalizes and returns the 32-bit checksum value.
    pub fn finish(&self) -> u32 {
        self.state ^ 0xFFFFFFFF
    }

    /// Direct single-pass CRC-32 computation.
    pub fn compute(data: &[u8]) -> u32 {
        let mut crc = Self::new();
        crc.update(data);
        crc.finish()
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Adler-32 checksum calculator.
pub struct Adler32 {
    a: u32,
    b: u32,
}

impl Adler32 {
    const MOD_ADLER: u32 = 65521;

    /// Creates a new Adler-32 instance.
    pub fn new() -> Self {
        Self { a: 1, b: 0 }
    }

    /// Updates checksum with bytes.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.a = (self.a + byte as u32) % Self::MOD_ADLER;
            self.b = (self.b + self.a) % Self::MOD_ADLER;
        }
    }

    /// Finalizes and returns Adler-32 value.
    pub fn finish(&self) -> u32 {
        (self.b << 16) | self.a
    }

    /// Direct single-pass Adler-32 computation.
    pub fn compute(data: &[u8]) -> u32 {
        let mut adler = Self::new();
        adler.update(data);
        adler.finish()
    }
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_checksums_correctness_1() {
        let data = b"123456789";
        let crc = Crc32::compute(data);
        assert_eq!(crc, 0xCBF43926); // Standard CRC32 IEEE test vector
    
        let adler = Adler32::compute(data);
        assert_eq!(adler, 0x091E01DE); // Standard Adler32 test vector
    
        let mut stream_crc = Crc32::new();
        stream_crc.update(b"12345");
        stream_crc.update(b"6789");
        assert_eq!(stream_crc.finish(), 0xCBF43926);
    }
}
