//! # Log Sinks
//!
//! Provides output destinations for log records: console, files with size rotation,
//! in-memory bounded ring buffers, and multi-sink fan-outs.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};
use crate::core::{UtilsError, UtilsResult};
use super::{LogLevel, LogRecord};

/// Log sink trait representing an output destination.
pub trait LogSink: Send + Sync {
    /// Writes a formatted log record string.
    fn write_record(&self, record: &LogRecord, formatted: &str) -> UtilsResult<()>;
    /// Flushes buffered output.
    fn flush(&self) -> UtilsResult<()>;
}

/// Standard console log sink with ANSI colors.
pub struct ConsoleSink {
    use_colors: bool,
}

impl ConsoleSink {
    /// Constructs a console sink.
    pub fn new(use_colors: bool) -> Self {
        Self { use_colors }
    }
}

impl LogSink for ConsoleSink {
    fn write_record(&self, record: &LogRecord, formatted: &str) -> UtilsResult<()> {
        if self.use_colors {
            let color_code = match record.level {
                LogLevel::Trace => "\x1b[36m", // Cyan
                LogLevel::Debug => "\x1b[34m", // Blue
                LogLevel::Info => "\x1b[32m",  // Green
                LogLevel::Warn => "\x1b[33m",  // Yellow
                LogLevel::Error => "\x1b[31m", // Red
                LogLevel::Off => "\x1b[0m",
            };
            let reset_code = "\x1b[0m";
            println!("{}{}{}", color_code, formatted, reset_code);
        } else {
            println!("{}", formatted);
        }
        Ok(())
    }

    fn flush(&self) -> UtilsResult<()> {
        std::io::stdout().flush().map_err(|e| UtilsError::IoError(e.to_string()))
    }
}

/// In-memory bounded ring buffer log sink.
#[derive(Debug, Clone)]
pub struct RingBufferSink {
    capacity: usize,
    buffer: Arc<Mutex<VecDeque<String>>>,
}

impl RingBufferSink {
    /// Constructs a ring buffer with maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
        }
    }

    /// Retrieves all current stored log messages.
    pub fn get_messages(&self) -> Vec<String> {
        if let Ok(b) = self.buffer.lock() {
            b.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Whether ring buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the ring buffer.
    pub fn clear(&self) {
        if let Ok(mut b) = self.buffer.lock() {
            b.clear();
        }
    }
}

impl LogSink for RingBufferSink {
    fn write_record(&self, _record: &LogRecord, formatted: &str) -> UtilsResult<()> {
        if let Ok(mut b) = self.buffer.lock() {
            if b.len() >= self.capacity {
                b.pop_front();
            }
            b.push_back(formatted.to_string());
        }
        Ok(())
    }

    fn flush(&self) -> UtilsResult<()> {
        Ok(())
    }
}

/// Fan-out sink writing to multiple sinks simultaneously.
pub struct MultiSink {
    sinks: Vec<Box<dyn LogSink>>,
}

impl MultiSink {
    /// Constructs a new MultiSink.
    pub fn new() -> Self {
        Self { sinks: Vec::new() }
    }

    /// Adds a sink destination.
    pub fn add_sink(&mut self, sink: Box<dyn LogSink>) -> &mut Self {
        self.sinks.push(sink);
        self
    }
}

impl Default for MultiSink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for MultiSink {
    fn write_record(&self, record: &LogRecord, formatted: &str) -> UtilsResult<()> {
        for s in &self.sinks {
            let _ = s.write_record(record, formatted);
        }
        Ok(())
    }

    fn flush(&self) -> UtilsResult<()> {
        for s in &self.sinks {
            let _ = s.flush();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_log_sinks_behavior_1() {
        let console = ConsoleSink::new(false);
        let rec = LogRecord::new(LogLevel::Info, "test", "hello world");
        assert!(console.write_record(&rec, "[INFO] test: hello world").is_ok());
        assert!(console.flush().is_ok());
    
        let ring = RingBufferSink::new(5);
        assert!(ring.is_empty());
        for k in 0..10 {
            let r = LogRecord::new(LogLevel::Debug, "mod", &format!("msg {}", k));
            assert!(ring.write_record(&r, &format!("entry {}", k)).is_ok());
        }
        assert_eq!(ring.len(), 5);
        let msgs = ring.get_messages();
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[4], "entry 9");
        ring.clear();
        assert_eq!(ring.len(), 0);
    
        let mut multi = MultiSink::new();
        multi.add_sink(Box::new(RingBufferSink::new(10)));
        assert!(multi.write_record(&rec, "multi msg").is_ok());
        assert!(multi.flush().is_ok());
    }
}
