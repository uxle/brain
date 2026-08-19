//! # Configuration Subsystem
//!
//! Provides multi-layered configuration management supporting defaults,
//! configuration files (JSON/INI/key-value), environment variables, and programmatic overrides.

pub mod schema;

use std::collections::BTreeMap;
use std::fmt;
use crate::core::UtilsResult;

/// Represents a source of configuration parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfigSource {
    /// Hardcoded system defaults (lowest precedence).
    Defaults = 0,
    /// System-wide configuration file.
    SystemFile = 1,
    /// User configuration file (e.g. ~/.config/brain.conf).
    UserFile = 2,
    /// Local project file (e.g. ./brain.json).
    ProjectFile = 3,
    /// Environment variables (e.g. BRAIN_*).
    Environment = 4,
    /// Programmatic CLI or API override (highest precedence).
    Override = 5,
}

impl fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Defaults => write!(f, "Defaults"),
            Self::SystemFile => write!(f, "SystemFile"),
            Self::UserFile => write!(f, "UserFile"),
            Self::ProjectFile => write!(f, "ProjectFile"),
            Self::Environment => write!(f, "Environment"),
            Self::Override => write!(f, "Override"),
        }
    }
}

/// A single configuration entry with source provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigEntry {
    /// String representation of value.
    pub value: String,
    /// The source that supplied this value.
    pub source: ConfigSource,
    /// Optional description or comment.
    pub description: Option<String>,
}

/// Multi-layered configuration dictionary.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConfigManager {
    entries: BTreeMap<String, ConfigEntry>,
}

impl ConfigManager {
    /// Creates an empty configuration manager.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Sets a key-value pair with a specific source level.
    /// Only updates if the new source precedence is >= existing source precedence.
    pub fn set(&mut self, key: &str, value: &str, source: ConfigSource) -> bool {
        if let Some(existing) = self.entries.get(key) {
            if source < existing.source {
                return false;
            }
        }
        self.entries.insert(
            key.to_string(),
            ConfigEntry {
                value: value.to_string(),
                source,
                description: None,
            },
        );
        true
    }

    /// Force sets a key-value pair regardless of precedence.
    pub fn force_set(&mut self, key: &str, value: &str, source: ConfigSource) {
        self.entries.insert(
            key.to_string(),
            ConfigEntry {
                value: value.to_string(),
                source,
                description: None,
            },
        );
    }

    /// Retrieves raw string value for key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|e| e.value.as_str())
    }

    /// Retrieves string value or default.
    pub fn get_str<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.get(key).unwrap_or(default)
    }

    /// Retrieves and parses integer value.
    pub fn get_i64(&self, key: &str, default: i64) -> i64 {
        self.get(key).and_then(|v| v.parse::<i64>().ok()).unwrap_or(default)
    }

    /// Retrieves and parses float value.
    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(|v| v.parse::<f64>().ok()).unwrap_or(default)
    }

    /// Retrieves and parses boolean value (accepts true, false, 1, 0, yes, no).
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.get(key).map(|s| s.to_lowercase()).as_deref() {
            Some("true") | Some("1") | Some("yes") | Some("on") => true,
            Some("false") | Some("0") | Some("no") | Some("off") => false,
            _ => default,
        }
    }

    /// Retrieves entry with source provenance.
    pub fn get_entry(&self, key: &str) -> Option<&ConfigEntry> {
        self.entries.get(key)
    }

    /// Checks if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of configured keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether configuration map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes a key.
    pub fn remove(&mut self, key: &str) -> Option<ConfigEntry> {
        self.entries.remove(key)
    }

    /// Returns iterator over keys and entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ConfigEntry)> {
        self.entries.iter()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Exports key-value map as INI/key-value text format.
    pub fn export_kv(&self) -> String {
        let mut out = String::new();
        for (k, entry) in &self.entries {
            out.push_str(&format!("{} = {}\n", k, entry.value));
        }
        out
    }

    /// Imports key-value pairs from text lines.
    pub fn import_kv(&mut self, text: &str, source: ConfigSource) -> UtilsResult<usize> {
        let mut count = 0;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
                continue;
            }
            if let Some((k, v)) = trimmed.split_once('=') {
                let k_trim = k.trim();
                let v_trim = v.trim().trim_matches('"').trim_matches('\'');
                if !k_trim.is_empty() {
                    self.set(k_trim, v_trim, source);
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_config_manager_precedence_1() {
        let mut cfg = ConfigManager::new();
        assert!(cfg.is_empty());
        
        assert!(cfg.set("threads", "4", ConfigSource::Defaults));
        assert_eq!(cfg.get_i64("threads", 1), 4);
        
        assert!(cfg.set("threads", "8", ConfigSource::ProjectFile));
        assert_eq!(cfg.get_i64("threads", 1), 8);
        
        assert!(!cfg.set("threads", "2", ConfigSource::Defaults));
        assert_eq!(cfg.get_i64("threads", 1), 8);
        
        assert!(cfg.set("threads", "16", ConfigSource::Environment));
        assert_eq!(cfg.get_i64("threads", 1), 16);
        
        cfg.set("verbose", "true", ConfigSource::Override);
        assert!(cfg.get_bool("verbose", false));
        cfg.set("debug", "off", ConfigSource::Override);
        assert!(!cfg.get_bool("debug", true));
        
        let text = cfg.export_kv();
        let mut imported = ConfigManager::new();
        let count = imported.import_kv(&text, ConfigSource::UserFile).unwrap();
        assert!(count >= 3);
        assert_eq!(imported.get_i64("threads", 1), 16);
    }
}
