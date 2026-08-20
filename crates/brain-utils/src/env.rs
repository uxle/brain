//! # Environment Utilities
//!
//! Provides typed environment variable access, prefix-scoped configuration
//! maps, boolean parsing, and default fallback mechanisms.

use std::collections::BTreeMap;
use std::env;

/// Retrieves an environment variable as a String, or None if absent.
pub fn env_get(key: &str) -> Option<String> {
    env::var(key).ok()
}

/// Retrieves an environment variable or returns default.
pub fn env_get_or(key: &str, default: &str) -> String {
    env_get(key).unwrap_or_else(|| default.to_string())
}

/// Retrieves and parses an integer environment variable.
pub fn env_i64(key: &str, default: i64) -> i64 {
    env_get(key)
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(default)
}

/// Retrieves and parses a float environment variable.
pub fn env_f64(key: &str, default: f64) -> f64 {
    env_get(key)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

/// Retrieves and parses a boolean environment variable.
pub fn env_bool(key: &str, default: bool) -> bool {
    match env_get(key).map(|s| s.to_lowercase()).as_deref() {
        Some("true") | Some("1") | Some("yes") | Some("on") => true,
        Some("false") | Some("0") | Some("no") | Some("off") => false,
        _ => default,
    }
}

/// Retrieves a comma-separated environment variable as a list of strings.
pub fn env_list(key: &str) -> Vec<String> {
    env_get(key)
        .map(|s| {
            s.split(',')
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Scans environment variables matching a given prefix (e.g., "BRAIN_").
/// Strips prefix and converts keys to lowercase.
pub fn env_prefix_map(prefix: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for (k, v) in env::vars() {
        if let Some(stripped) = k.strip_prefix(prefix) {
            map.insert(stripped.to_lowercase(), v);
        }
    }
    map
}

/// Environment configuration holder.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvConfig {
    /// Prefix to search for (e.g. "BRAIN_").
    pub prefix: String,
    /// Cached variables.
    pub vars: BTreeMap<String, String>,
}

impl EnvConfig {
    /// Loads all environment variables with specified prefix.
    pub fn from_prefix(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            vars: env_prefix_map(prefix),
        }
    }

    /// Gets a value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.vars.get(&key.to_lowercase()).map(|s| s.as_str())
    }

    /// Gets integer value.
    pub fn get_i64(&self, key: &str, default: i64) -> i64 {
        self.get(key)
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(default)
    }

    /// Gets boolean value.
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.get(key).map(|s| s.to_lowercase()).as_deref() {
            Some("true") | Some("1") | Some("yes") | Some("on") => true,
            Some("false") | Some("0") | Some("no") | Some("off") => false,
            _ => default,
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_env_utilities_1() {
        let dummy_key = format!("BRAIN_TEST_VAR_{}", 1);
        assert_eq!(env_get_or(&dummy_key, "fallback"), "fallback");
        assert_eq!(env_i64(&dummy_key, 42), 42);
        assert_eq!(env_f64(&dummy_key, 3.14), 3.14);
        assert!(!env_bool(&dummy_key, false));

        let list = env_list(&dummy_key);
        assert!(list.is_empty());

        let env_cfg = EnvConfig::from_prefix("BRAIN_");
        assert_eq!(env_cfg.prefix, "BRAIN_");
    }
}
