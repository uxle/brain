//! # Formatting & Conversion Utilities
//!
//! Formatting helpers for durations, throughputs, FLOPs, memory sizes, and string parsing.

use std::time::Duration;

/// Formats duration in nanoseconds to a human-readable string with appropriate units.
pub fn format_duration(nanos: f64) -> String {
    if nanos < 1_000.0 {
        format!("{:.2} ns", nanos)
    } else if nanos < 1_000_000.0 {
        format!("{:.2} µs", nanos / 1_000.0)
    } else if nanos < 1_000_000_000.0 {
        format!("{:.2} ms", nanos / 1_000_000.0)
    } else {
        format!("{:.2} s", nanos / 1_000_000_000.0)
    }
}

/// Formats bytes per second throughput into B/s, KB/s, MB/s, or GB/s.
pub fn format_throughput(bytes_per_sec: f64) -> String {
    if bytes_per_sec < 1_024.0 {
        format!("{:.2} B/s", bytes_per_sec)
    } else if bytes_per_sec < 1_048_576.0 {
        format!("{:.2} KB/s", bytes_per_sec / 1_024.0)
    } else if bytes_per_sec < 1_073_741_824.0 {
        format!("{:.2} MB/s", bytes_per_sec / 1_048_576.0)
    } else {
        format!("{:.2} GB/s", bytes_per_sec / 1_073_741_824.0)
    }
}

/// Formats floating-point operations per second into MFLOPS, GFLOPS, or TFLOPS.
pub fn format_gflops(gflops: f64) -> String {
    if gflops < 1.0 {
        format!("{:.2} MFLOPS", gflops * 1000.0)
    } else if gflops < 1000.0 {
        format!("{:.2} GFLOPS", gflops)
    } else {
        format!("{:.2} TFLOPS", gflops / 1000.0)
    }
}

/// Parses a human-readable size string (e.g. `"64KB"`, `"2MB"`, `"1GB"`) into bytes.
pub fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim().to_uppercase();
    if s.ends_with("GB") {
        let val: f64 = s[..s.len() - 2].trim().parse().ok()?;
        Some((val * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if s.ends_with("MB") {
        let val: f64 = s[..s.len() - 2].trim().parse().ok()?;
        Some((val * 1024.0 * 1024.0) as usize)
    } else if s.ends_with("KB") {
        let val: f64 = s[..s.len() - 2].trim().parse().ok()?;
        Some((val * 1024.0) as usize)
    } else if s.ends_with('B') {
        s[..s.len() - 1].trim().parse().ok()
    } else {
        s.parse().ok()
    }
}

/// Parses a duration string (e.g. `"100ns"`, `"50us"`, `"10ms"`, `"2s"`) into a `Duration`.
pub fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim().to_lowercase();
    if s.ends_with("ns") {
        let val: u64 = s[..s.len() - 2].trim().parse().ok()?;
        Some(Duration::from_nanos(val))
    } else if s.ends_with("us") || s.ends_with("µs") {
        let val: u64 = s[..s.len() - 2].trim().parse().ok()?;
        Some(Duration::from_micros(val))
    } else if s.ends_with("ms") {
        let val: u64 = s[..s.len() - 2].trim().parse().ok()?;
        Some(Duration::from_millis(val))
    } else if s.ends_with('s') {
        let val: f64 = s[..s.len() - 1].trim().parse().ok()?;
        Some(Duration::from_secs_f64(val))
    } else {
        None
    }
}

/// Rounds a floating-point value to the specified number of decimal places.
pub fn round_places(val: f64, places: usize) -> f64 {
    let factor = 10.0_f64.powi(places as i32);
    (val * factor).round() / factor
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
