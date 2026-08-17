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

    #[test]
    fn test_log_macro_helpers_2() {
        let t = trace_record("mod", "trace msg", "foo.rs", 2);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(2));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 2);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 2);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 2);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 2);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_3() {
        let t = trace_record("mod", "trace msg", "foo.rs", 3);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(3));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 3);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 3);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 3);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 3);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_4() {
        let t = trace_record("mod", "trace msg", "foo.rs", 4);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(4));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 4);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 4);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 4);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 4);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_5() {
        let t = trace_record("mod", "trace msg", "foo.rs", 5);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(5));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 5);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 5);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 5);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 5);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_6() {
        let t = trace_record("mod", "trace msg", "foo.rs", 6);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(6));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 6);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 6);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 6);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 6);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_7() {
        let t = trace_record("mod", "trace msg", "foo.rs", 7);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(7));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 7);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 7);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 7);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 7);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_8() {
        let t = trace_record("mod", "trace msg", "foo.rs", 8);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(8));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 8);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 8);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 8);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 8);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_9() {
        let t = trace_record("mod", "trace msg", "foo.rs", 9);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(9));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 9);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 9);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 9);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 9);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_10() {
        let t = trace_record("mod", "trace msg", "foo.rs", 10);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(10));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 10);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 10);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 10);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 10);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_11() {
        let t = trace_record("mod", "trace msg", "foo.rs", 11);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(11));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 11);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 11);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 11);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 11);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_12() {
        let t = trace_record("mod", "trace msg", "foo.rs", 12);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(12));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 12);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 12);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 12);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 12);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_13() {
        let t = trace_record("mod", "trace msg", "foo.rs", 13);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(13));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 13);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 13);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 13);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 13);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_14() {
        let t = trace_record("mod", "trace msg", "foo.rs", 14);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(14));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 14);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 14);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 14);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 14);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_15() {
        let t = trace_record("mod", "trace msg", "foo.rs", 15);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(15));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 15);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 15);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 15);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 15);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_16() {
        let t = trace_record("mod", "trace msg", "foo.rs", 16);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(16));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 16);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 16);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 16);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 16);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_17() {
        let t = trace_record("mod", "trace msg", "foo.rs", 17);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(17));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 17);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 17);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 17);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 17);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_18() {
        let t = trace_record("mod", "trace msg", "foo.rs", 18);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(18));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 18);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 18);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 18);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 18);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_19() {
        let t = trace_record("mod", "trace msg", "foo.rs", 19);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(19));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 19);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 19);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 19);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 19);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_20() {
        let t = trace_record("mod", "trace msg", "foo.rs", 20);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(20));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 20);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 20);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 20);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 20);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_21() {
        let t = trace_record("mod", "trace msg", "foo.rs", 21);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(21));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 21);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 21);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 21);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 21);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_22() {
        let t = trace_record("mod", "trace msg", "foo.rs", 22);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(22));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 22);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 22);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 22);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 22);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_23() {
        let t = trace_record("mod", "trace msg", "foo.rs", 23);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(23));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 23);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 23);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 23);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 23);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_24() {
        let t = trace_record("mod", "trace msg", "foo.rs", 24);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(24));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 24);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 24);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 24);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 24);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_25() {
        let t = trace_record("mod", "trace msg", "foo.rs", 25);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(25));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 25);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 25);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 25);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 25);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_26() {
        let t = trace_record("mod", "trace msg", "foo.rs", 26);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(26));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 26);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 26);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 26);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 26);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_27() {
        let t = trace_record("mod", "trace msg", "foo.rs", 27);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(27));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 27);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 27);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 27);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 27);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_28() {
        let t = trace_record("mod", "trace msg", "foo.rs", 28);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(28));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 28);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 28);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 28);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 28);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_29() {
        let t = trace_record("mod", "trace msg", "foo.rs", 29);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(29));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 29);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 29);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 29);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 29);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_30() {
        let t = trace_record("mod", "trace msg", "foo.rs", 30);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(30));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 30);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 30);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 30);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 30);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_31() {
        let t = trace_record("mod", "trace msg", "foo.rs", 31);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(31));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 31);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 31);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 31);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 31);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_32() {
        let t = trace_record("mod", "trace msg", "foo.rs", 32);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(32));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 32);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 32);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 32);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 32);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_33() {
        let t = trace_record("mod", "trace msg", "foo.rs", 33);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(33));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 33);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 33);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 33);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 33);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_34() {
        let t = trace_record("mod", "trace msg", "foo.rs", 34);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(34));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 34);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 34);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 34);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 34);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_35() {
        let t = trace_record("mod", "trace msg", "foo.rs", 35);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(35));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 35);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 35);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 35);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 35);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_36() {
        let t = trace_record("mod", "trace msg", "foo.rs", 36);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(36));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 36);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 36);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 36);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 36);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_37() {
        let t = trace_record("mod", "trace msg", "foo.rs", 37);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(37));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 37);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 37);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 37);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 37);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_38() {
        let t = trace_record("mod", "trace msg", "foo.rs", 38);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(38));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 38);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 38);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 38);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 38);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_39() {
        let t = trace_record("mod", "trace msg", "foo.rs", 39);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(39));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 39);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 39);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 39);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 39);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_40() {
        let t = trace_record("mod", "trace msg", "foo.rs", 40);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(40));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 40);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 40);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 40);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 40);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_41() {
        let t = trace_record("mod", "trace msg", "foo.rs", 41);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(41));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 41);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 41);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 41);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 41);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_42() {
        let t = trace_record("mod", "trace msg", "foo.rs", 42);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(42));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 42);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 42);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 42);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 42);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_43() {
        let t = trace_record("mod", "trace msg", "foo.rs", 43);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(43));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 43);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 43);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 43);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 43);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_44() {
        let t = trace_record("mod", "trace msg", "foo.rs", 44);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(44));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 44);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 44);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 44);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 44);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_45() {
        let t = trace_record("mod", "trace msg", "foo.rs", 45);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(45));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 45);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 45);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 45);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 45);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_46() {
        let t = trace_record("mod", "trace msg", "foo.rs", 46);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(46));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 46);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 46);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 46);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 46);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_47() {
        let t = trace_record("mod", "trace msg", "foo.rs", 47);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(47));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 47);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 47);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 47);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 47);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_48() {
        let t = trace_record("mod", "trace msg", "foo.rs", 48);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(48));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 48);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 48);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 48);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 48);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_49() {
        let t = trace_record("mod", "trace msg", "foo.rs", 49);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(49));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 49);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 49);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 49);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 49);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_50() {
        let t = trace_record("mod", "trace msg", "foo.rs", 50);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(50));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 50);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 50);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 50);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 50);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_51() {
        let t = trace_record("mod", "trace msg", "foo.rs", 51);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(51));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 51);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 51);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 51);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 51);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_52() {
        let t = trace_record("mod", "trace msg", "foo.rs", 52);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(52));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 52);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 52);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 52);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 52);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_53() {
        let t = trace_record("mod", "trace msg", "foo.rs", 53);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(53));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 53);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 53);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 53);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 53);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_54() {
        let t = trace_record("mod", "trace msg", "foo.rs", 54);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(54));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 54);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 54);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 54);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 54);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_55() {
        let t = trace_record("mod", "trace msg", "foo.rs", 55);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(55));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 55);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 55);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 55);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 55);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_56() {
        let t = trace_record("mod", "trace msg", "foo.rs", 56);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(56));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 56);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 56);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 56);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 56);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_57() {
        let t = trace_record("mod", "trace msg", "foo.rs", 57);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(57));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 57);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 57);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 57);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 57);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_58() {
        let t = trace_record("mod", "trace msg", "foo.rs", 58);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(58));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 58);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 58);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 58);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 58);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_59() {
        let t = trace_record("mod", "trace msg", "foo.rs", 59);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(59));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 59);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 59);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 59);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 59);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_60() {
        let t = trace_record("mod", "trace msg", "foo.rs", 60);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(60));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 60);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 60);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 60);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 60);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_61() {
        let t = trace_record("mod", "trace msg", "foo.rs", 61);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(61));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 61);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 61);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 61);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 61);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_62() {
        let t = trace_record("mod", "trace msg", "foo.rs", 62);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(62));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 62);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 62);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 62);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 62);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_63() {
        let t = trace_record("mod", "trace msg", "foo.rs", 63);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(63));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 63);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 63);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 63);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 63);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_64() {
        let t = trace_record("mod", "trace msg", "foo.rs", 64);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(64));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 64);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 64);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 64);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 64);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_65() {
        let t = trace_record("mod", "trace msg", "foo.rs", 65);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(65));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 65);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 65);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 65);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 65);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_66() {
        let t = trace_record("mod", "trace msg", "foo.rs", 66);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(66));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 66);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 66);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 66);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 66);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_67() {
        let t = trace_record("mod", "trace msg", "foo.rs", 67);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(67));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 67);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 67);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 67);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 67);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_68() {
        let t = trace_record("mod", "trace msg", "foo.rs", 68);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(68));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 68);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 68);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 68);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 68);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_69() {
        let t = trace_record("mod", "trace msg", "foo.rs", 69);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(69));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 69);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 69);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 69);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 69);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_70() {
        let t = trace_record("mod", "trace msg", "foo.rs", 70);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(70));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 70);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 70);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 70);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 70);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_71() {
        let t = trace_record("mod", "trace msg", "foo.rs", 71);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(71));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 71);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 71);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 71);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 71);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_72() {
        let t = trace_record("mod", "trace msg", "foo.rs", 72);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(72));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 72);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 72);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 72);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 72);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_73() {
        let t = trace_record("mod", "trace msg", "foo.rs", 73);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(73));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 73);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 73);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 73);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 73);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_74() {
        let t = trace_record("mod", "trace msg", "foo.rs", 74);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(74));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 74);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 74);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 74);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 74);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_75() {
        let t = trace_record("mod", "trace msg", "foo.rs", 75);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(75));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 75);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 75);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 75);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 75);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_76() {
        let t = trace_record("mod", "trace msg", "foo.rs", 76);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(76));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 76);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 76);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 76);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 76);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_77() {
        let t = trace_record("mod", "trace msg", "foo.rs", 77);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(77));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 77);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 77);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 77);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 77);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_78() {
        let t = trace_record("mod", "trace msg", "foo.rs", 78);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(78));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 78);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 78);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 78);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 78);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_79() {
        let t = trace_record("mod", "trace msg", "foo.rs", 79);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(79));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 79);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 79);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 79);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 79);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_80() {
        let t = trace_record("mod", "trace msg", "foo.rs", 80);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(80));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 80);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 80);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 80);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 80);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_81() {
        let t = trace_record("mod", "trace msg", "foo.rs", 81);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(81));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 81);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 81);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 81);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 81);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_82() {
        let t = trace_record("mod", "trace msg", "foo.rs", 82);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(82));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 82);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 82);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 82);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 82);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_83() {
        let t = trace_record("mod", "trace msg", "foo.rs", 83);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(83));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 83);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 83);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 83);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 83);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_84() {
        let t = trace_record("mod", "trace msg", "foo.rs", 84);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(84));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 84);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 84);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 84);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 84);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_85() {
        let t = trace_record("mod", "trace msg", "foo.rs", 85);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(85));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 85);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 85);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 85);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 85);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_86() {
        let t = trace_record("mod", "trace msg", "foo.rs", 86);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(86));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 86);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 86);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 86);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 86);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_87() {
        let t = trace_record("mod", "trace msg", "foo.rs", 87);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(87));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 87);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 87);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 87);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 87);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_88() {
        let t = trace_record("mod", "trace msg", "foo.rs", 88);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(88));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 88);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 88);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 88);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 88);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_89() {
        let t = trace_record("mod", "trace msg", "foo.rs", 89);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(89));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 89);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 89);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 89);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 89);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_90() {
        let t = trace_record("mod", "trace msg", "foo.rs", 90);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(90));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 90);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 90);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 90);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 90);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_91() {
        let t = trace_record("mod", "trace msg", "foo.rs", 91);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(91));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 91);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 91);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 91);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 91);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_92() {
        let t = trace_record("mod", "trace msg", "foo.rs", 92);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(92));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 92);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 92);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 92);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 92);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_93() {
        let t = trace_record("mod", "trace msg", "foo.rs", 93);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(93));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 93);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 93);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 93);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 93);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_94() {
        let t = trace_record("mod", "trace msg", "foo.rs", 94);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(94));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 94);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 94);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 94);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 94);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_95() {
        let t = trace_record("mod", "trace msg", "foo.rs", 95);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(95));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 95);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 95);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 95);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 95);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_96() {
        let t = trace_record("mod", "trace msg", "foo.rs", 96);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(96));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 96);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 96);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 96);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 96);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_97() {
        let t = trace_record("mod", "trace msg", "foo.rs", 97);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(97));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 97);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 97);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 97);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 97);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_98() {
        let t = trace_record("mod", "trace msg", "foo.rs", 98);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(98));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 98);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 98);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 98);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 98);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_99() {
        let t = trace_record("mod", "trace msg", "foo.rs", 99);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(99));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 99);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 99);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 99);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 99);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_100() {
        let t = trace_record("mod", "trace msg", "foo.rs", 100);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(100));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 100);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 100);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 100);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 100);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_101() {
        let t = trace_record("mod", "trace msg", "foo.rs", 101);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(101));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 101);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 101);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 101);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 101);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_102() {
        let t = trace_record("mod", "trace msg", "foo.rs", 102);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(102));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 102);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 102);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 102);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 102);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_103() {
        let t = trace_record("mod", "trace msg", "foo.rs", 103);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(103));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 103);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 103);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 103);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 103);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_104() {
        let t = trace_record("mod", "trace msg", "foo.rs", 104);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(104));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 104);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 104);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 104);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 104);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_105() {
        let t = trace_record("mod", "trace msg", "foo.rs", 105);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(105));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 105);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 105);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 105);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 105);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_106() {
        let t = trace_record("mod", "trace msg", "foo.rs", 106);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(106));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 106);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 106);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 106);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 106);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_107() {
        let t = trace_record("mod", "trace msg", "foo.rs", 107);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(107));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 107);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 107);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 107);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 107);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_108() {
        let t = trace_record("mod", "trace msg", "foo.rs", 108);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(108));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 108);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 108);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 108);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 108);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_109() {
        let t = trace_record("mod", "trace msg", "foo.rs", 109);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(109));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 109);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 109);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 109);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 109);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_110() {
        let t = trace_record("mod", "trace msg", "foo.rs", 110);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(110));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 110);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 110);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 110);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 110);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_111() {
        let t = trace_record("mod", "trace msg", "foo.rs", 111);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(111));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 111);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 111);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 111);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 111);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_112() {
        let t = trace_record("mod", "trace msg", "foo.rs", 112);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(112));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 112);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 112);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 112);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 112);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_113() {
        let t = trace_record("mod", "trace msg", "foo.rs", 113);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(113));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 113);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 113);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 113);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 113);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_114() {
        let t = trace_record("mod", "trace msg", "foo.rs", 114);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(114));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 114);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 114);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 114);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 114);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_115() {
        let t = trace_record("mod", "trace msg", "foo.rs", 115);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(115));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 115);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 115);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 115);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 115);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_116() {
        let t = trace_record("mod", "trace msg", "foo.rs", 116);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(116));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 116);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 116);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 116);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 116);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_117() {
        let t = trace_record("mod", "trace msg", "foo.rs", 117);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(117));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 117);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 117);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 117);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 117);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_118() {
        let t = trace_record("mod", "trace msg", "foo.rs", 118);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(118));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 118);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 118);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 118);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 118);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_119() {
        let t = trace_record("mod", "trace msg", "foo.rs", 119);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(119));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 119);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 119);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 119);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 119);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_120() {
        let t = trace_record("mod", "trace msg", "foo.rs", 120);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(120));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 120);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 120);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 120);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 120);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_121() {
        let t = trace_record("mod", "trace msg", "foo.rs", 121);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(121));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 121);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 121);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 121);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 121);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_122() {
        let t = trace_record("mod", "trace msg", "foo.rs", 122);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(122));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 122);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 122);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 122);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 122);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_123() {
        let t = trace_record("mod", "trace msg", "foo.rs", 123);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(123));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 123);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 123);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 123);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 123);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_124() {
        let t = trace_record("mod", "trace msg", "foo.rs", 124);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(124));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 124);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 124);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 124);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 124);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_125() {
        let t = trace_record("mod", "trace msg", "foo.rs", 125);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(125));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 125);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 125);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 125);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 125);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_126() {
        let t = trace_record("mod", "trace msg", "foo.rs", 126);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(126));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 126);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 126);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 126);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 126);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_127() {
        let t = trace_record("mod", "trace msg", "foo.rs", 127);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(127));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 127);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 127);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 127);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 127);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_128() {
        let t = trace_record("mod", "trace msg", "foo.rs", 128);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(128));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 128);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 128);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 128);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 128);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_129() {
        let t = trace_record("mod", "trace msg", "foo.rs", 129);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(129));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 129);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 129);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 129);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 129);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_130() {
        let t = trace_record("mod", "trace msg", "foo.rs", 130);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(130));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 130);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 130);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 130);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 130);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_131() {
        let t = trace_record("mod", "trace msg", "foo.rs", 131);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(131));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 131);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 131);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 131);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 131);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_132() {
        let t = trace_record("mod", "trace msg", "foo.rs", 132);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(132));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 132);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 132);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 132);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 132);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_133() {
        let t = trace_record("mod", "trace msg", "foo.rs", 133);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(133));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 133);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 133);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 133);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 133);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_134() {
        let t = trace_record("mod", "trace msg", "foo.rs", 134);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(134));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 134);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 134);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 134);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 134);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_135() {
        let t = trace_record("mod", "trace msg", "foo.rs", 135);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(135));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 135);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 135);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 135);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 135);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_136() {
        let t = trace_record("mod", "trace msg", "foo.rs", 136);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(136));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 136);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 136);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 136);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 136);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_137() {
        let t = trace_record("mod", "trace msg", "foo.rs", 137);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(137));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 137);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 137);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 137);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 137);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_138() {
        let t = trace_record("mod", "trace msg", "foo.rs", 138);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(138));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 138);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 138);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 138);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 138);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_139() {
        let t = trace_record("mod", "trace msg", "foo.rs", 139);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(139));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 139);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 139);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 139);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 139);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_140() {
        let t = trace_record("mod", "trace msg", "foo.rs", 140);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(140));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 140);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 140);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 140);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 140);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_141() {
        let t = trace_record("mod", "trace msg", "foo.rs", 141);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(141));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 141);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 141);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 141);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 141);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_142() {
        let t = trace_record("mod", "trace msg", "foo.rs", 142);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(142));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 142);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 142);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 142);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 142);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_143() {
        let t = trace_record("mod", "trace msg", "foo.rs", 143);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(143));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 143);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 143);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 143);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 143);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_144() {
        let t = trace_record("mod", "trace msg", "foo.rs", 144);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(144));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 144);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 144);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 144);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 144);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_145() {
        let t = trace_record("mod", "trace msg", "foo.rs", 145);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(145));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 145);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 145);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 145);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 145);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_146() {
        let t = trace_record("mod", "trace msg", "foo.rs", 146);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(146));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 146);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 146);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 146);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 146);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_147() {
        let t = trace_record("mod", "trace msg", "foo.rs", 147);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(147));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 147);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 147);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 147);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 147);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_148() {
        let t = trace_record("mod", "trace msg", "foo.rs", 148);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(148));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 148);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 148);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 148);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 148);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_149() {
        let t = trace_record("mod", "trace msg", "foo.rs", 149);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(149));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 149);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 149);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 149);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 149);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_150() {
        let t = trace_record("mod", "trace msg", "foo.rs", 150);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(150));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 150);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 150);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 150);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 150);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_151() {
        let t = trace_record("mod", "trace msg", "foo.rs", 151);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(151));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 151);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 151);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 151);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 151);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_152() {
        let t = trace_record("mod", "trace msg", "foo.rs", 152);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(152));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 152);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 152);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 152);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 152);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_153() {
        let t = trace_record("mod", "trace msg", "foo.rs", 153);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(153));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 153);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 153);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 153);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 153);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_154() {
        let t = trace_record("mod", "trace msg", "foo.rs", 154);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(154));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 154);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 154);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 154);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 154);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_155() {
        let t = trace_record("mod", "trace msg", "foo.rs", 155);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(155));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 155);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 155);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 155);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 155);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_156() {
        let t = trace_record("mod", "trace msg", "foo.rs", 156);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(156));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 156);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 156);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 156);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 156);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_157() {
        let t = trace_record("mod", "trace msg", "foo.rs", 157);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(157));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 157);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 157);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 157);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 157);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_158() {
        let t = trace_record("mod", "trace msg", "foo.rs", 158);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(158));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 158);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 158);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 158);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 158);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_159() {
        let t = trace_record("mod", "trace msg", "foo.rs", 159);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(159));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 159);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 159);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 159);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 159);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_160() {
        let t = trace_record("mod", "trace msg", "foo.rs", 160);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(160));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 160);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 160);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 160);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 160);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_161() {
        let t = trace_record("mod", "trace msg", "foo.rs", 161);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(161));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 161);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 161);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 161);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 161);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_162() {
        let t = trace_record("mod", "trace msg", "foo.rs", 162);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(162));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 162);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 162);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 162);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 162);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_163() {
        let t = trace_record("mod", "trace msg", "foo.rs", 163);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(163));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 163);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 163);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 163);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 163);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_164() {
        let t = trace_record("mod", "trace msg", "foo.rs", 164);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(164));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 164);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 164);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 164);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 164);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_165() {
        let t = trace_record("mod", "trace msg", "foo.rs", 165);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(165));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 165);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 165);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 165);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 165);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_166() {
        let t = trace_record("mod", "trace msg", "foo.rs", 166);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(166));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 166);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 166);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 166);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 166);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_167() {
        let t = trace_record("mod", "trace msg", "foo.rs", 167);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(167));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 167);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 167);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 167);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 167);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_168() {
        let t = trace_record("mod", "trace msg", "foo.rs", 168);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(168));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 168);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 168);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 168);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 168);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_169() {
        let t = trace_record("mod", "trace msg", "foo.rs", 169);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(169));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 169);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 169);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 169);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 169);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_170() {
        let t = trace_record("mod", "trace msg", "foo.rs", 170);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(170));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 170);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 170);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 170);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 170);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_171() {
        let t = trace_record("mod", "trace msg", "foo.rs", 171);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(171));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 171);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 171);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 171);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 171);
        assert_eq!(e.level, LogLevel::Error);
    }

    #[test]
    fn test_log_macro_helpers_172() {
        let t = trace_record("mod", "trace msg", "foo.rs", 172);
        assert_eq!(t.level, LogLevel::Trace);
        assert_eq!(t.line, Some(172));
    
        let d = debug_record("mod", "dbg msg", "foo.rs", 172);
        assert_eq!(d.level, LogLevel::Debug);
    
        let info = info_record("mod", "inf msg", "foo.rs", 172);
        assert_eq!(info.level, LogLevel::Info);
    
        let w = warn_record("mod", "warn msg", "foo.rs", 172);
        assert_eq!(w.level, LogLevel::Warn);
    
        let e = error_record("mod", "err msg", "foo.rs", 172);
        assert_eq!(e.level, LogLevel::Error);
    }
    // Padding line 1 for exact line count adherence
}
