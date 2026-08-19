//! # Logging Framework
//!
//! Production-grade logging framework featuring log levels, record metadata,
//! thread-safe sinks, structured formatters, and global level filtering.

pub mod sinks;
pub mod formatters;
pub mod macros;

use std::fmt;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};
use self::sinks::{ConsoleSink, LogSink};
use self::formatters::{LogFormatter, PlainFormatter};

/// Logging verbosity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Finest diagnostics for debugging.
    Trace = 0,
    /// Detailed diagnostic information.
    Debug = 1,
    /// Normal operational messages.
    Info = 2,
    /// Warning of potential issues.
    Warn = 3,
    /// Error conditions requiring attention.
    Error = 4,
    /// Logging disabled.
    Off = 5,
}

impl LogLevel {
    /// Parses a string into a LogLevel.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_uppercase().as_str() {
            "TRACE" => Self::Trace,
            "DEBUG" => Self::Debug,
            "INFO" => Self::Info,
            "WARN" | "WARNING" => Self::Warn,
            "ERROR" => Self::Error,
            "OFF" | "NONE" => Self::Off,
            _ => Self::Info,
        }
    }

    /// Converts to string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
            Self::Off => "OFF",
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A structured record representing a single log event.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRecord {
    /// Severity level.
    pub level: LogLevel,
    /// Log message text.
    pub message: String,
    /// Target module or component name.
    pub target: String,
    /// Source file name.
    pub file: Option<String>,
    /// Source line number.
    pub line: Option<u32>,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Key-value metadata tags.
    pub fields: Vec<(String, String)>,
}

impl LogRecord {
    /// Creates a new log record.
    pub fn new(level: LogLevel, target: &str, message: &str) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            level,
            message: message.to_string(),
            target: target.to_string(),
            file: None,
            line: None,
            timestamp_ms: ts,
            fields: Vec::new(),
        }
    }

    /// Attaches file and line info.
    pub fn with_location(mut self, file: &str, line: u32) -> Self {
        self.file = Some(file.to_string());
        self.line = Some(line);
        self
    }

    /// Attaches a key-value attribute.
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }
}

/// Core logger trait.
pub trait Logger: Send + Sync {
    /// Logs a record if the level meets threshold.
    fn log(&self, record: &LogRecord);
    /// Gets current level filter.
    fn level(&self) -> LogLevel;
    /// Sets level filter.
    fn set_level(&self, level: LogLevel);
    /// Flushes any buffered records.
    fn flush(&self);
}

/// Standard thread-safe logger implementation.
pub struct StandardLogger {
    level: RwLock<LogLevel>,
    sink: Box<dyn LogSink>,
    formatter: Box<dyn LogFormatter>,
}

impl StandardLogger {
    /// Constructs a standard logger with a sink and formatter.
    pub fn new(level: LogLevel, sink: Box<dyn LogSink>, formatter: Box<dyn LogFormatter>) -> Self {
        Self {
            level: RwLock::new(level),
            sink,
            formatter,
        }
    }

    /// Creates a default console logger at INFO level.
    pub fn default_console() -> Self {
        Self::new(
            LogLevel::Info,
            Box::new(ConsoleSink::new(true)),
            Box::new(PlainFormatter::new()),
        )
    }
}

impl Logger for StandardLogger {
    fn log(&self, record: &LogRecord) {
        let cur_level = self.level.read().map(|l| *l).unwrap_or(LogLevel::Info);
        if record.level >= cur_level && cur_level != LogLevel::Off {
            let formatted = self.formatter.format(record);
            let _ = self.sink.write_record(record, &formatted);
        }
    }

    fn level(&self) -> LogLevel {
        self.level.read().map(|l| *l).unwrap_or(LogLevel::Info)
    }

    fn set_level(&self, level: LogLevel) {
        if let Ok(mut w) = self.level.write() {
            *w = level;
        }
    }

    fn flush(&self) {
        let _ = self.sink.flush();
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_log_levels_and_records_1() {
        assert_eq!(LogLevel::from_str("debug"), LogLevel::Debug);
        assert_eq!(LogLevel::from_str("INFO"), LogLevel::Info);
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
    
        let rec = LogRecord::new(LogLevel::Warn, "engine", "test message")
            .with_location("main.rs", 1)
            .with_field("batch", "1");
    
        assert_eq!(rec.level, LogLevel::Warn);
        assert_eq!(rec.target, "engine");
        assert_eq!(rec.message, "test message");
        assert_eq!(rec.line, Some(1));
        assert_eq!(rec.fields.len(), 1);
    
        let logger = StandardLogger::default_console();
        assert_eq!(logger.level(), LogLevel::Info);
        logger.set_level(LogLevel::Debug);
        assert_eq!(logger.level(), LogLevel::Debug);
        logger.log(&rec);
        logger.flush();
    }
}
