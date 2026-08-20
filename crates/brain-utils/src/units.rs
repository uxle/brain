//! # Unit Formatting and Parsing
//!
//! Provides formatting and parsing for human-readable byte sizes,
//! durations, frequencies, and percentages.

use crate::core::{UtilsError, UtilsResult};
use std::time::Duration;

/// Formats a byte count into binary units (B, KiB, MiB, GiB, TiB).
pub fn format_bytes_binary(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    const TIB: u64 = GIB * 1024;

    if bytes >= TIB {
        format!("{:.2} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Formats a byte count into decimal units (B, KB, MB, GB, TB).
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1000;
    const MB: u64 = KB * 1000;
    const GB: u64 = MB * 1000;
    const TB: u64 = GB * 1000;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Parses human readable size string (e.g. "512MB", "1.5 GiB") into bytes.
pub fn parse_size(s: &str) -> UtilsResult<u64> {
    let trimmed = s.trim().to_uppercase();
    if trimmed.ends_with("TIB") {
        let num = trimmed[..trimmed.len() - 3]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1024.0 * 1024.0 * 1024.0 * 1024.0) as u64)
    } else if trimmed.ends_with("GIB") {
        let num = trimmed[..trimmed.len() - 3]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1024.0 * 1024.0 * 1024.0) as u64)
    } else if trimmed.ends_with("MIB") {
        let num = trimmed[..trimmed.len() - 3]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1024.0 * 1024.0) as u64)
    } else if trimmed.ends_with("KIB") {
        let num = trimmed[..trimmed.len() - 3]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1024.0) as u64)
    } else if trimmed.ends_with("TB") {
        let num = trimmed[..trimmed.len() - 2]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1e12) as u64)
    } else if trimmed.ends_with("GB") {
        let num = trimmed[..trimmed.len() - 2]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1e9) as u64)
    } else if trimmed.ends_with("MB") {
        let num = trimmed[..trimmed.len() - 2]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1e6) as u64)
    } else if trimmed.ends_with("KB") {
        let num = trimmed[..trimmed.len() - 2]
            .trim()
            .parse::<f64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))?;
        Ok((num * 1e3) as u64)
    } else if trimmed.ends_with('B') {
        trimmed[..trimmed.len() - 1]
            .trim()
            .parse::<u64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))
    } else {
        trimmed
            .parse::<u64>()
            .map_err(|e| UtilsError::ParseError(e.to_string()))
    }
}

/// Formats a Duration into compact readable text.
pub fn format_duration(dur: Duration) -> String {
    let nanos = dur.as_nanos();
    if nanos < 1_000 {
        format!("{} ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.2} µs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else {
        let secs = dur.as_secs_f64();
        if secs < 60.0 {
            format!("{:.2} s", secs)
        } else if secs < 3600.0 {
            format!("{:.2} min", secs / 60.0)
        } else {
            format!("{:.2} h", secs / 3600.0)
        }
    }
}

/// Formats a float ratio (0.0..1.0) as percentage string.
pub fn format_percent(ratio: f64) -> String {
    format!("{:.2}%", ratio * 100.0)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_units_formatting_and_parsing_1() {
        assert_eq!(format_bytes(1000), "1.00 KB");
        assert_eq!(format_bytes_binary(1024), "1.00 KiB");
        assert_eq!(format_bytes_binary(512), "512 B");

        assert_eq!(parse_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_size("10MB").unwrap(), 10_000_000);
        assert_eq!(parse_size("4096B").unwrap(), 4096);

        let dur = Duration::from_millis(250);
        assert_eq!(format_duration(dur), "250.00 ms");
        assert_eq!(format_percent(0.854), "85.40%");
    }
}
