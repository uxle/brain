//! # INI / TOML-Lite Parser and Serializer
//!
//! Provides parsing and generation of INI configuration files supporting
//! sections, comments, key-value pairs, and typed value conversions.

use std::collections::BTreeMap;
use crate::core::UtilsResult;

/// INI configuration file AST representation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IniFile {
    sections: BTreeMap<String, BTreeMap<String, String>>,
}

impl IniFile {
    /// Creates an empty INI file container.
    pub fn new() -> Self {
        Self {
            sections: BTreeMap::new(),
        }
    }

    /// Sets a key-value pair under a section (use "" or "default" for global).
    pub fn set(&mut self, section: &str, key: &str, value: &str) {
        self.sections
            .entry(section.to_string())
            .or_default()
            .insert(key.to_string(), value.to_string());
    }

    /// Retrieves string value.
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.sections.get(section).and_then(|sec| sec.get(key).map(|s| s.as_str()))
    }

    /// Retrieves and parses integer value.
    pub fn get_i64(&self, section: &str, key: &str, default: i64) -> i64 {
        self.get(section, key).and_then(|v| v.parse::<i64>().ok()).unwrap_or(default)
    }

    /// Retrieves and parses float value.
    pub fn get_f64(&self, section: &str, key: &str, default: f64) -> f64 {
        self.get(section, key).and_then(|v| v.parse::<f64>().ok()).unwrap_or(default)
    }

    /// Retrieves and parses boolean value.
    pub fn get_bool(&self, section: &str, key: &str, default: bool) -> bool {
        match self.get(section, key).map(|s| s.to_lowercase()).as_deref() {
            Some("true") | Some("1") | Some("yes") | Some("on") => true,
            Some("false") | Some("0") | Some("no") | Some("off") => false,
            _ => default,
        }
    }

    /// Exports INI structure to formatted text string.
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        for (sec_name, sec_map) in &self.sections {
            if !sec_name.is_empty() {
                out.push_str(&format!("[{}]\n", sec_name));
            }
            for (k, v) in sec_map {
                out.push_str(&format!("{} = {}\n", k, v));
            }
            out.push('\n');
        }
        out
    }

    /// Parses INI text into an IniFile structure.
    pub fn parse(text: &str) -> UtilsResult<Self> {
        let mut ini = Self::new();
        let mut cur_section = String::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
                continue;
            }
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                cur_section = trimmed[1..trimmed.len() - 1].trim().to_string();
            } else if let Some((k, v)) = trimmed.split_once('=') {
                let key = k.trim().to_string();
                let val = v.trim().trim_matches('"').trim_matches('\'').to_string();
                if !key.is_empty() {
                    ini.set(&cur_section, &key, &val);
                }
            }
        }
        Ok(ini)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_ini_parser_and_sections_1() {
        let ini_text = "[server]\nhost = 127.0.0.1\nport = 8080\nenabled = true\n\n[model]\nlayers = 24\n";
        let ini = IniFile::parse(ini_text).unwrap();
    
        assert_eq!(ini.get("server", "host"), Some("127.0.0.1"));
        assert_eq!(ini.get_i64("server", "port", 80), 8080);
        assert!(ini.get_bool("server", "enabled", false));
        assert_eq!(ini.get_i64("model", "layers", 12), 24);
    
        let serialized = ini.serialize();
        assert!(serialized.contains("[server]"));
        assert!(serialized.contains("port = 8080"));
    }
}
