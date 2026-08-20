//! # Optimizer State & Checkpointing
//!
//! State dictionary management, buffer persistence, metadata tracking, and checkpoint serialization.
#![allow(missing_docs)]

use brain_core::Tensor;
use std::collections::HashMap;

/// Metadata stored alongside optimizer state.
#[derive(Debug, Clone, PartialEq)]
pub struct StateMetadata {
    pub step: usize,
    pub optimizer_type: String,
    pub version: String,
    pub timestamp: u64,
    pub num_param_groups: usize,
}

impl Default for StateMetadata {
    fn default() -> Self {
        Self {
            step: 0,
            optimizer_type: "Unknown".to_string(),
            version: "0.2.0".to_string(),
            timestamp: 0,
            num_param_groups: 1,
        }
    }
}

/// A comprehensive state dictionary containing tensors and metadata.
#[derive(Debug, Clone, Default)]
pub struct StateDict {
    pub metadata: StateMetadata,
    pub tensors: HashMap<String, Tensor>,
    pub scalars: HashMap<String, f64>,
}

impl StateDict {
    /// Creates a new empty state dictionary.
    pub fn new(optimizer_type: impl Into<String>, step: usize) -> Self {
        Self {
            metadata: StateMetadata {
                step,
                optimizer_type: optimizer_type.into(),
                version: "0.2.0".to_string(),
                timestamp: 0,
                num_param_groups: 1,
            },
            tensors: HashMap::new(),
            scalars: HashMap::new(),
        }
    }

    /// Inserts a tensor buffer.
    pub fn insert_tensor(&mut self, key: impl Into<String>, tensor: Tensor) {
        self.tensors.insert(key.into(), tensor);
    }

    /// Inserts a scalar value.
    pub fn insert_scalar(&mut self, key: impl Into<String>, value: f64) {
        self.scalars.insert(key.into(), value);
    }

    /// Retrieves a tensor buffer reference.
    pub fn get_tensor(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }

    /// Retrieves a scalar value.
    pub fn get_scalar(&self, key: &str) -> Option<f64> {
        self.scalars.get(key).copied()
    }

    /// Returns the total number of tensor buffers in state.
    pub fn num_buffers(&self) -> usize {
        self.tensors.len()
    }

    /// Total number of stored scalar values.
    pub fn num_scalars(&self) -> usize {
        self.scalars.len()
    }

    /// Serializes state dictionary into deterministic text-encoded bytes.
    pub fn save_bytes(&self) -> Vec<u8> {
        let mut out = format!(
            "OPTIM_STATE_V1|{}|{}|{}|{}\n",
            self.metadata.optimizer_type,
            self.metadata.step,
            self.metadata.version,
            self.metadata.num_param_groups
        );
        for (k, v) in &self.scalars {
            out.push_str(&format!("scalar|{}|{}\n", k, v));
        }
        for (k, t) in &self.tensors {
            let shape_str = t
                .shape()
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("x");
            let data_str = t
                .data()
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!("tensor|{}|{}|{}\n", k, shape_str, data_str));
        }
        out.into_bytes()
    }

    /// Deserializes state dictionary from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        let mut lines = text.lines();
        let header = lines.next().ok_or_else(|| "empty bytes".to_string())?;
        let h_parts: Vec<&str> = header.split('|').collect();
        if h_parts.len() < 5 || h_parts[0] != "OPTIM_STATE_V1" {
            return Err("invalid header".to_string());
        }
        let optimizer_type = h_parts[1].to_string();
        let step = h_parts[2].parse::<usize>().map_err(|e| e.to_string())?;
        let version = h_parts[3].to_string();
        let num_param_groups = h_parts[4].parse::<usize>().unwrap_or(1);

        let mut sd = StateDict {
            metadata: StateMetadata {
                step,
                optimizer_type,
                version,
                timestamp: 0,
                num_param_groups,
            },
            tensors: HashMap::new(),
            scalars: HashMap::new(),
        };

        for line in lines {
            let parts: Vec<&str> = line.split('|').collect();
            match parts.as_slice() {
                ["scalar", k, v] => {
                    let val = v.parse::<f64>().map_err(|e| e.to_string())?;
                    sd.insert_scalar(*k, val);
                }
                ["tensor", k, s, d] => {
                    let shape: Vec<usize> = if s.is_empty() {
                        Vec::new()
                    } else {
                        s.split('x')
                            .map(|p| p.parse::<usize>().map_err(|e| e.to_string()))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                    let data: Vec<f64> = if d.is_empty() {
                        Vec::new()
                    } else {
                        d.split(',')
                            .map(|p| p.parse::<f64>().map_err(|e| e.to_string()))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                    sd.insert_tensor(*k, Tensor::from_vec(data, shape));
                }
                _ => {}
            }
        }
        Ok(sd)
    }
}

/// Checkpoint manager for saving and restoring optimizer state dictionaries.
#[derive(Debug, Clone, Default)]
pub struct OptimizerCheckpoint {
    pub state_dict: StateDict,
}

impl OptimizerCheckpoint {
    pub fn from_state_dict(state_dict: StateDict) -> Self {
        Self { state_dict }
    }

    pub fn into_state_dict(self) -> StateDict {
        self.state_dict
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
