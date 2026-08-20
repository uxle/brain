//! # Configuration File Parser & Precedence Hierarchy
//!
//! Loads and merges settings from `~/.brainrc`, `.brain.toml`, environment variables, and CLI flags.

use std::collections::HashMap;

/// Parsed configuration file representation.
#[derive(Debug, Clone, Default)]
pub struct ConfigFile {
    values: HashMap<String, String>,
    sections: HashMap<String, HashMap<String, String>>,
}

impl ConfigFile {
    /// Creates an empty `ConfigFile`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses configuration content in TOML-like key-value format.
    pub fn parse(content: &str) -> Self {
        let mut values = HashMap::new();
        let mut sections = HashMap::new();
        let mut current_section: Option<String> = None;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
                continue;
            }

            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                let sec_name = trimmed[1..trimmed.len() - 1].trim().to_string();
                current_section = Some(sec_name);
                continue;
            }

            if let Some(pos) = trimmed.find('=') {
                let key = trimmed[..pos].trim().to_string();
                let mut val = trimmed[pos + 1..].trim();
                if (val.starts_with('"') && val.ends_with('"'))
                    || (val.starts_with('\'') && val.ends_with('\''))
                {
                    val = &val[1..val.len() - 1];
                }

                if let Some(sec) = &current_section {
                    sections
                        .entry(sec.clone())
                        .or_insert_with(HashMap::new)
                        .insert(key, val.to_string());
                } else {
                    values.insert(key, val.to_string());
                }
            }
        }

        Self { values, sections }
    }

    /// Gets a root property value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    /// Gets a section property value.
    pub fn get_section(&self, section: &str, key: &str) -> Option<&str> {
        self.sections.get(section)?.get(key).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
