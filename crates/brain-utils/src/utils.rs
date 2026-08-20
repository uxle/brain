//! # General Utility Functions
//!
//! Provides high-precision time measurements, random UUID generation,
//! shell parameter quoting, string sanitation, and terminal text styling.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Returns current Unix epoch timestamp in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Returns current Unix epoch timestamp in microseconds.
pub fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Returns current Unix epoch timestamp in nanoseconds.
pub fn now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Computes elapsed time since an earlier Instant.
pub fn elapsed_since(start: Instant) -> Duration {
    start.elapsed()
}

/// Fast pseudo-random number generator for UUID and token generation.
#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    /// Creates a seeded FastRng.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    /// Creates a FastRng seeded from system time.
    pub fn from_time() -> Self {
        Self::new(now_ns() as u64)
    }

    /// Generates next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generates next pseudo-random u32.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() & 0xFFFFFFFF) as u32
    }

    /// Generates next float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Generates a random v4-compliant UUID string without external dependencies.
pub fn random_uuid_lite() -> String {
    let mut rng = FastRng::from_time();
    let p1 = rng.next_u32();
    let p2 = (rng.next_u32() & 0x0FFF) | 0x4000; // v4 version
    let p3 = (rng.next_u32() & 0x3FFF) | 0x8000; // variant
    let p4 = rng.next_u64() & 0xFFFFFFFFFFFF;

    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        p1,
        (p2 >> 16),
        p2 & 0xFFFF,
        p3,
        p4
    )
}

/// Safely quotes an argument for POSIX shell execution.
pub fn shell_quote(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_string();
    }
    if arg
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | '=' | ':' | '@'))
    {
        return arg.to_string();
    }
    let mut escaped = String::from("'");
    for c in arg.chars() {
        if c == '\'' {
            escaped.push_str("'\\''");
        } else {
            escaped.push(c);
        }
    }
    escaped.push('\'');
    escaped
}

/// Sanitizes a string for use as a filesystem path.
pub fn sanitize_filename(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out
    }
}

/// Truncates string with ellipsis if exceeding max length.
pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        let mut out: String = s.chars().take(max_len - 3).collect();
        out.push_str("...");
        out
    }
}

/// Pads string to the right up to width with pad char.
pub fn pad_right(s: &str, width: usize, pad: char) -> String {
    let count = s.chars().count();
    if count >= width {
        s.to_string()
    } else {
        let mut res = s.to_string();
        for _ in 0..(width - count) {
            res.push(pad);
        }
        res
    }
}

/// Pads string to the left up to width with pad char.
pub fn pad_left(s: &str, width: usize, pad: char) -> String {
    let count = s.chars().count();
    if count >= width {
        s.to_string()
    } else {
        let mut res = String::new();
        for _ in 0..(width - count) {
            res.push(pad);
        }
        res.push_str(s);
        res
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_general_utils_helpers_1() {
        let t_ms = now_ms();
        let t_us = now_us();
        assert!(t_us >= t_ms * 1000 || t_ms > 0);

        let uuid = random_uuid_lite();
        assert_eq!(uuid.len(), 36);
        assert_eq!(uuid.matches('-').count(), 4);

        let safe = shell_quote("simple_arg_123");
        assert_eq!(safe, "simple_arg_123");
        let unsafe_arg = shell_quote("hello world; rm -rf /");
        assert!(unsafe_arg.starts_with("'"));

        let clean = sanitize_filename("bad/file:name*?.txt");
        assert_eq!(clean, "bad_file_name__.txt");

        let trunc = truncate_str("superlongstringexample", 10);
        assert_eq!(trunc, "superlo...");

        let left = pad_left("42", 5, '0');
        assert_eq!(left, "00042");
        let right = pad_right("hi", 5, '.');
        assert_eq!(right, "hi...");
    }
}
