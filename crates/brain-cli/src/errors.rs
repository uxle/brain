//! # Diagnostic Error Formatting & Suggestion Hints
//!
//! Generates user-facing diagnostic error messages with context and suggestion hints.

use crate::core::ExitCode;

/// Represents a formatted CLI error condition.
#[derive(Debug, Clone)]
pub struct CliError {
    pub message: String,
    pub suggestions: Vec<String>,
    pub exit_code: ExitCode,
}

impl CliError {
    /// Creates a new `CliError` with the default error exit code.
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            suggestions: Vec::new(),
            exit_code: ExitCode::ERROR,
        }
    }

    /// Adds a suggestion hint.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Formats the error for console display.
    pub fn format_report(&self) -> String {
        let mut out = format!("error: {}\n", self.message);
        for s in &self.suggestions {
            out.push_str(&format!("  hint: {}\n", s));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
