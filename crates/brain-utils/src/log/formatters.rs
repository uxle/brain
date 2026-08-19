//! # Log Formatters
//!
//! Provides formatting strategies for log records: Plain text, Key=Value,
//! JSON structured lines, and customizable token templates.

use super::LogRecord;

/// Formatter trait converting a log record into a display string.
pub trait LogFormatter: Send + Sync {
    /// Formats a record into a string.
    fn format(&self, record: &LogRecord) -> String;
}

/// Plain text formatter: `[LEVEL] [target] message`.
#[derive(Debug, Clone, Default)]
pub struct PlainFormatter;

impl PlainFormatter {
    /// Creates a new PlainFormatter.
    pub fn new() -> Self {
        Self
    }
}

impl LogFormatter for PlainFormatter {
    fn format(&self, record: &LogRecord) -> String {
        format!("[{}] [{}] {}", record.level, record.target, record.message)
    }
}

/// Timestamped formatter: `2026-08-17 12:00:00 [LEVEL] [target] (file:line) message`.
#[derive(Debug, Clone, Default)]
pub struct TimestampedFormatter {
    include_location: bool,
}

impl TimestampedFormatter {
    /// Creates a timestamped formatter.
    pub fn new(include_location: bool) -> Self {
        Self { include_location }
    }
}

impl LogFormatter for TimestampedFormatter {
    fn format(&self, record: &LogRecord) -> String {
        let loc = if self.include_location {
            match (&record.file, record.line) {
                (Some(f), Some(l)) => format!(" ({}:{})", f, l),
                _ => String::new(),
            }
        } else {
            String::new()
        };
        format!("[ts:{}] [{}] [{}]{} {}", record.timestamp_ms, record.level, record.target, loc, record.message)
    }
}

/// Key=Value formatter: `level=INFO target=engine msg="hello" k1=v1`.
#[derive(Debug, Clone, Default)]
pub struct KeyValueFormatter;

impl KeyValueFormatter {
    /// Creates a KeyValueFormatter.
    pub fn new() -> Self {
        Self
    }
}

impl LogFormatter for KeyValueFormatter {
    fn format(&self, record: &LogRecord) -> String {
        let mut out = format!(
            "time_ms={} level={} target={} msg=\"{}\"",
            record.timestamp_ms,
            record.level,
            record.target,
            record.message.replace('"', "\\\"")
        );
        for (k, v) in &record.fields {
            out.push_str(&format!(" {}=\"{}\"", k, v.replace('"', "\\\"")));
        }
        out
    }
}

/// JSON formatter: `{"timestamp_ms":123,"level":"INFO","target":"engine","message":"..."}`.
#[derive(Debug, Clone, Default)]
pub struct JsonFormatter;

impl JsonFormatter {
    /// Creates a JsonFormatter.
    pub fn new() -> Self {
        Self
    }
}

impl LogFormatter for JsonFormatter {
    fn format(&self, record: &LogRecord) -> String {
        let mut json = format!(
            "{{\"timestamp_ms\":{},\"level\":\"{}\",\"target\":\"{}\",\"message\":\"{}\"",
            record.timestamp_ms,
            record.level,
            record.target,
            record.message.replace('"', "\\\"").replace('\n', "\\n")
        );
        if let Some(ref f) = record.file {
            json.push_str(&format!(",\"file\":\"{}\"", f));
        }
        if let Some(l) = record.line {
            json.push_str(&format!(",\"line\":{}", l));
        }
        if !record.fields.is_empty() {
            json.push_str(",\"fields\":{");
            for (i, (k, v)) in record.fields.iter().enumerate() {
                if i > 0 {
                    json.push(',');
                }
                json.push_str(&format!("\"{}\":\"{}\"", k, v.replace('"', "\\\"")));
            }
            json.push('}');
        }
        json.push('}');
        json
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_formatters_output_1() {
        let rec = LogRecord::new(crate::log::LogLevel::Info, "network", "packet received")
            .with_location("net.rs", 1)
            .with_field("bytes", "64");
    
        let plain = PlainFormatter::new().format(&rec);
        assert_eq!(plain, "[INFO] [network] packet received");
    
        let ts = TimestampedFormatter::new(true).format(&rec);
        assert!(ts.contains("[INFO]"));
        assert!(ts.contains("net.rs:"));
    
        let kv = KeyValueFormatter::new().format(&rec);
        assert!(kv.contains("level=INFO"));
        assert!(kv.contains("bytes="));
    
        let json = JsonFormatter::new().format(&rec);
        assert!(json.contains("target"));
        assert!(json.contains("network"));
    }
}
