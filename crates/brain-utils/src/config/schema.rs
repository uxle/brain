//! # Configuration Schema Validation
//!
//! Provides schema definition, type checking, range validation, and constraint
//! enforcement for configuration parameters.

use super::ConfigManager;
use crate::core::{UtilsError, UtilsResult};
use std::collections::BTreeMap;

/// Supported schema field data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// String data.
    String,
    /// 64-bit signed integer.
    Integer,
    /// 64-bit floating point number.
    Float,
    /// Boolean flag.
    Boolean,
    /// Comma-separated list of strings.
    List,
}

/// Validation constraint on a configuration value.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// Minimum numerical value (inclusive).
    MinFloat(f64),
    /// Maximum numerical value (inclusive).
    MaxFloat(f64),
    /// Minimum integer value (inclusive).
    MinInt(i64),
    /// Maximum integer value (inclusive).
    MaxInt(i64),
    /// Value must be one of allowed strings.
    OneOf(Vec<String>),
    /// Minimum string length.
    MinLength(usize),
    /// Maximum string length.
    MaxLength(usize),
    /// String must not be empty or whitespace.
    NonEmpty,
}

/// Field definition within a schema.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    /// Configuration key name.
    pub name: String,
    /// Expected data type.
    pub field_type: FieldType,
    /// Whether key is strictly required.
    pub required: bool,
    /// Default value if omitted.
    pub default_value: Option<String>,
    /// List of constraints.
    pub constraints: Vec<Constraint>,
    /// Human-readable documentation.
    pub description: String,
}

impl FieldDef {
    /// Constructs a required field.
    pub fn required(name: &str, field_type: FieldType) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required: true,
            default_value: None,
            constraints: Vec::new(),
            description: String::new(),
        }
    }

    /// Constructs an optional field with default.
    pub fn optional(name: &str, field_type: FieldType, default: &str) -> Self {
        Self {
            name: name.to_string(),
            field_type,
            required: false,
            default_value: Some(default.to_string()),
            constraints: Vec::new(),
            description: String::new(),
        }
    }

    /// Adds a constraint.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Sets description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }
}

/// Validator for enforcing configuration schemas.
#[derive(Debug, Clone, Default)]
pub struct SchemaValidator {
    fields: BTreeMap<String, FieldDef>,
}

impl SchemaValidator {
    /// Constructs a new schema validator.
    pub fn new() -> Self {
        Self {
            fields: BTreeMap::new(),
        }
    }

    /// Registers a field definition.
    pub fn add_field(&mut self, field: FieldDef) -> &mut Self {
        self.fields.insert(field.name.clone(), field);
        self
    }

    /// Validates a configuration manager against this schema.
    pub fn validate(&self, config: &ConfigManager) -> UtilsResult<()> {
        for (name, field) in &self.fields {
            match config.get(name) {
                Some(val) => {
                    self.validate_value(field, val)?;
                }
                None => {
                    if field.required {
                        return Err(UtilsError::ValidationError(format!(
                            "Missing required configuration key: '{}'",
                            name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_value(&self, field: &FieldDef, val: &str) -> UtilsResult<()> {
        match field.field_type {
            FieldType::String => {}
            FieldType::Integer => {
                let parsed = val.parse::<i64>().map_err(|_| {
                    UtilsError::ValidationError(format!(
                        "Key '{}' expected integer, found '{}'",
                        field.name, val
                    ))
                })?;
                for c in &field.constraints {
                    if let Constraint::MinInt(min) = c {
                        if parsed < *min {
                            return Err(UtilsError::ValidationError(format!(
                                "Key '{}' value {} below minimum {}",
                                field.name, parsed, min
                            )));
                        }
                    }
                    if let Constraint::MaxInt(max) = c {
                        if parsed > *max {
                            return Err(UtilsError::ValidationError(format!(
                                "Key '{}' value {} exceeds maximum {}",
                                field.name, parsed, max
                            )));
                        }
                    }
                }
            }
            FieldType::Float => {
                let parsed = val.parse::<f64>().map_err(|_| {
                    UtilsError::ValidationError(format!(
                        "Key '{}' expected float, found '{}'",
                        field.name, val
                    ))
                })?;
                for c in &field.constraints {
                    if let Constraint::MinFloat(min) = c {
                        if parsed < *min {
                            return Err(UtilsError::ValidationError(format!(
                                "Key '{}' value {} below minimum {}",
                                field.name, parsed, min
                            )));
                        }
                    }
                    if let Constraint::MaxFloat(max) = c {
                        if parsed > *max {
                            return Err(UtilsError::ValidationError(format!(
                                "Key '{}' value {} exceeds maximum {}",
                                field.name, parsed, max
                            )));
                        }
                    }
                }
            }
            FieldType::Boolean => {
                let lower = val.to_lowercase();
                if !["true", "false", "1", "0", "yes", "no", "on", "off"].contains(&lower.as_str())
                {
                    return Err(UtilsError::ValidationError(format!(
                        "Key '{}' expected boolean flag, found '{}'",
                        field.name, val
                    )));
                }
            }
            FieldType::List => {}
        }

        for c in &field.constraints {
            match c {
                Constraint::OneOf(allowed) => {
                    if !allowed.contains(&val.to_string()) {
                        return Err(UtilsError::ValidationError(format!(
                            "Key '{}' value '{}' not in allowed set: {:?}",
                            field.name, val, allowed
                        )));
                    }
                }
                Constraint::MinLength(min_len) => {
                    if val.len() < *min_len {
                        return Err(UtilsError::ValidationError(format!(
                            "Key '{}' length {} shorter than minimum {}",
                            field.name,
                            val.len(),
                            min_len
                        )));
                    }
                }
                Constraint::MaxLength(max_len) => {
                    if val.len() > *max_len {
                        return Err(UtilsError::ValidationError(format!(
                            "Key '{}' length {} exceeds maximum {}",
                            field.name,
                            val.len(),
                            max_len
                        )));
                    }
                }
                Constraint::NonEmpty if val.trim().is_empty() => {
                    return Err(UtilsError::ValidationError(format!(
                        "Key '{}' cannot be empty",
                        field.name
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_schema_validator_1() {
        use crate::config::ConfigSource;
        let mut schema = SchemaValidator::new();
        schema.add_field(
            FieldDef::required("learning_rate", FieldType::Float)
                .with_constraint(Constraint::MinFloat(0.00001))
                .with_constraint(Constraint::MaxFloat(1.0)),
        );
        schema.add_field(
            FieldDef::required("batch_size", FieldType::Integer)
                .with_constraint(Constraint::MinInt(1))
                .with_constraint(Constraint::MaxInt(4096)),
        );
        schema.add_field(
            FieldDef::optional("optimizer", FieldType::String, "adam").with_constraint(
                Constraint::OneOf(vec!["adam".into(), "sgd".into(), "adamw".into()]),
            ),
        );

        let mut cfg = ConfigManager::new();
        cfg.set("learning_rate", "0.001", ConfigSource::Defaults);
        cfg.set("batch_size", "32", ConfigSource::Defaults);
        cfg.set("optimizer", "adamw", ConfigSource::Defaults);

        assert!(schema.validate(&cfg).is_ok());

        cfg.set("batch_size", "0", ConfigSource::Override);
        assert!(schema.validate(&cfg).is_err());
    }
}
