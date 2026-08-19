//! # Logging Macros and Helpers
//!
//! Provides ergonomic macros and structured logging dispatch functions.

use super::{LogLevel, LogRecord};

/// Constructs a log record with level, target, and message.
pub fn create_record(level: LogLevel, target: &str, msg: &str, file: &str, line: u32) -> LogRecord {
    LogRecord::new(level, target, msg).with_location(file, line)
}

/// Formats and returns a TRACE level log record.
pub fn trace_record(target: &str, msg: &str, file: &str, line: u32) -> LogRecord {
    create_record(LogLevel::Trace, target, msg, file, line)
}

/// Formats and returns a DEBUG level log record.
pub fn debug_record(target: &str, msg: &str, file: &str, line: u32) -> LogRecord {
    create_record(LogLevel::Debug, target, msg, file, line)
}

/// Formats and returns an INFO level log record.
pub fn info_record(target: &str, msg: &str, file: &str, line: u32) -> LogRecord {
    create_record(LogLevel::Info, target, msg, file, line)
}

/// Formats and returns a WARN level log record.
pub fn warn_record(target: &str, msg: &str, file: &str, line: u32) -> LogRecord {
    create_record(LogLevel::Warn, target, msg, file, line)
}

/// Formats and returns an ERROR level log record.
pub fn error_record(target: &str, msg: &str, file: &str, line: u32) -> LogRecord {
    create_record(LogLevel::Error, target, msg, file, line)
}

/// Macro for trace logging.
#[macro_export]
macro_rules! log_trace {
    ($target:expr, $($arg:tt)*) => {
        $crate::log::macros::trace_record($target, &format!($($arg)*), file!(), line!())
    };
}

/// Macro for debug logging.
#[macro_export]
macro_rules! log_debug {
    ($target:expr, $($arg:tt)*) => {
        $crate::log::macros::debug_record($target, &format!($($arg)*), file!(), line!())
    };
}

/// Macro for info logging.
#[macro_export]
macro_rules! log_info {
    ($target:expr, $($arg:tt)*) => {
        $crate::log::macros::info_record($target, &format!($($arg)*), file!(), line!())
    };
}

/// Macro for warn logging.
#[macro_export]
macro_rules! log_warn {
    ($target:expr, $($arg:tt)*) => {
        $crate::log::macros::warn_record($target, &format!($($arg)*), file!(), line!())
    };
}

/// Macro for error logging.
#[macro_export]
macro_rules! log_error {
    ($target:expr, $($arg:tt)*) => {
        $crate::log::macros::error_record($target, &format!($($arg)*), file!(), line!())
    };
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_log_macro_helpers_1() {
        let t = trace_record("mod", "trace msg", "foo.rs", 1);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(1));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 1);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 1);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 1);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 1);
        assert_eq!(e.level, LogLevel::Error);
    }
}
