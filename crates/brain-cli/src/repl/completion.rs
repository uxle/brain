//! # REPL Tab-Completion & Fuzzy Matching
//!
//! Provides interactive auto-completion for REPL commands, math functions, and session variables.

/// Completer for REPL prompts.
#[derive(Default)]
pub struct ReplCompleter {
    builtins: Vec<String>,
}

impl ReplCompleter {
    /// Creates a new `ReplCompleter`.
    pub fn new() -> Self {
        Self {
            builtins: vec![
                ":help".to_string(),
                ":vars".to_string(),
                ":clear".to_string(),
                ":quit".to_string(),
                "zeros".to_string(),
                "ones".to_string(),
                "matmul".to_string(),
                "sin".to_string(),
                "exp".to_string(),
            ],
        }
    }

    /// Finds completion suggestions starting with `prefix`.
    pub fn complete(&self, prefix: &str, var_names: &[&str]) -> Vec<String> {
        let mut results = Vec::new();
        for b in &self.builtins {
            if b.starts_with(prefix) {
                results.push(b.clone());
            }
        }
        for v in var_names {
            if v.starts_with(prefix) {
                results.push(v.to_string());
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
