//! # Core CLI Types, Output Sinks & Exit Codes
//!
//! Provides the primary primitives for CLI command execution, exit codes,
//! formatted output serialization, and in-memory or standard stream output sinks.

use std::sync::{Arc, Mutex};

/// Process exit codes for command executions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExitCode(pub i32);

impl ExitCode {
    pub const SUCCESS: Self = Self(0);
    pub const ERROR: Self = Self(1);
    pub const INVALID_USAGE: Self = Self(2);
    pub const IO_ERROR: Self = Self(3);
    pub const NOT_FOUND: Self = Self(4);
    pub const INTERRUPTED: Self = Self(130);

    /// Returns whether the exit code represents a success.
    pub fn is_success(&self) -> bool {
        self.0 == 0
    }
}

/// Output serialization formats for CLI command responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
    Yaml,
}

/// Destination sink for standard output and diagnostics.
#[derive(Clone)]
pub struct OutputSink {
    buffer: Option<Arc<Mutex<String>>>,
}

impl Default for OutputSink {
    fn default() -> Self {
        Self::stdout()
    }
}

impl OutputSink {
    /// Creates a sink directing output to process standard out.
    pub fn stdout() -> Self {
        Self { buffer: None }
    }

    /// Creates an in-memory capturing sink for testing and programmatic capture.
    pub fn memory() -> Self {
        Self {
            buffer: Some(Arc::new(Mutex::new(String::new()))),
        }
    }

    /// Writes a line of text to the sink.
    pub fn println(&self, msg: &str) {
        if let Some(buf) = &self.buffer {
            let mut b = buf.lock().unwrap();
            b.push_str(msg);
            b.push('\n');
        } else {
            println!("{}", msg);
        }
    }

    /// Writes unformatted text to the sink without a trailing newline.
    pub fn print(&self, msg: &str) {
        if let Some(buf) = &self.buffer {
            let mut b = buf.lock().unwrap();
            b.push_str(msg);
        } else {
            print!("{}", msg);
        }
    }

    /// Retrieves captured text if operating in memory mode.
    pub fn captured(&self) -> Option<String> {
        self.buffer.as_ref().map(|b| b.lock().unwrap().clone())
    }
}

/// Specification of a registered CLI command.
#[derive(Debug, Clone)]
pub struct CommandSpec {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub aliases: Vec<String>,
}

impl CommandSpec {
    /// Creates a new `CommandSpec`.
    pub fn new(name: impl Into<String>, desc: impl Into<String>, usage: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: desc.into(),
            usage: usage.into(),
            aliases: Vec::new(),
        }
    }

    /// Adds an alias name.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
