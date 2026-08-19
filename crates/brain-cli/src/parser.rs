//! # Command Line Argument & Flag Parser
//!
//! Provides flexible CLI argument parsing: long/short flags (`-v`, `--verbose`),
//! key-value pairs (`--lr=0.01`), `--no-X` negations, positionals, and typo suggestions.

use std::collections::{HashMap, HashSet};

/// Parsed result of command-line arguments.
#[derive(Debug, Clone, Default)]
pub struct ArgMatches {
    pub subcommand: Option<String>,
    pub positionals: Vec<String>,
    pub flags: HashSet<String>,
    pub options: HashMap<String, Vec<String>>,
}

impl ArgMatches {
    /// Returns whether the given boolean flag was provided.
    pub fn has_flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    /// Gets the single value for the given option key.
    pub fn get_option(&self, name: &str) -> Option<&str> {
        self.options.get(name)?.last().map(|s| s.as_str())
    }

    /// Gets all values for a repeatable option key.
    pub fn get_all_options(&self, name: &str) -> Option<&[String]> {
        self.options.get(name).map(|v| v.as_slice())
    }

    /// Gets the positional argument at index `idx`.
    pub fn get_positional(&self, idx: usize) -> Option<&str> {
        self.positionals.get(idx).map(|s| s.as_str())
    }
}

/// Argument parser with flag extraction.
#[derive(Default)]
pub struct ArgParser {
    allowed_flags: HashSet<String>,
    allowed_options: HashSet<String>,
}

impl ArgParser {
    /// Creates a new `ArgParser`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares an allowed boolean flag.
    pub fn flag(mut self, name: impl Into<String>) -> Self {
        self.allowed_flags.insert(name.into());
        self
    }

    /// Declares an allowed key-value option.
    pub fn option(mut self, name: impl Into<String>) -> Self {
        self.allowed_options.insert(name.into());
        self
    }

    /// Parses a slice of argument strings.
    pub fn parse(&self, args: &[String]) -> Result<ArgMatches, String> {
        let mut matches = ArgMatches::default();
        let mut i = 0;

        while i < args.len() {
            let arg = &args[i];
            if let Some(name) = arg.strip_prefix("--") {
                if let Some(eq_pos) = name.find('=') {
                    let opt_name = &name[..eq_pos];
                    let opt_val = &name[eq_pos + 1..];
                    matches.options.entry(opt_name.to_string()).or_default().push(opt_val.to_string());
                } else if let Some(stripped) = name.strip_prefix("no-") {
                    matches.flags.remove(stripped);
                } else if self.allowed_options.contains(name) {
                    let next_is_value = if let Some(next) = args.get(i + 1) {
                        !next.starts_with('-') || next.parse::<f64>().is_ok()
                    } else {
                        false
                    };
                    if next_is_value {
                        i += 1;
                        matches.options.entry(name.to_string()).or_default().push(args[i].clone());
                    } else {
                        return Err(format!("Option '--{}' requires a value", name));
                    }
                } else {
                    matches.flags.insert(name.to_string());
                }
            } else if let Some(short_name) = arg.strip_prefix('-') {
                if !short_name.is_empty() {
                    matches.flags.insert(short_name.to_string());
                }
            } else {
                matches.positionals.push(arg.clone());
            }
            i += 1;
        }

        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
