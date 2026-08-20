//! # Host CPU Memory Activation Offloading
//!
//! Offloads large intermediate tensors from high-pressure accelerator memory to CPU host RAM.

use brain_core::{BrainError, BrainResult, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

/// Manages offloading and prefetching of activation tensors.
#[derive(Default)]
pub struct CpuOffloader {
    storage: Mutex<HashMap<usize, Tensor>>,
}

impl CpuOffloader {
    /// Creates a new `CpuOffloader`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Offloads a tensor identified by `tensor_id`.
    pub fn offload(&self, tensor_id: usize, tensor: Tensor) -> BrainResult<()> {
        let mut guard = self.storage.lock().unwrap();
        guard.insert(tensor_id, tensor);
        Ok(())
    }

    /// Restores a previously offloaded tensor.
    pub fn restore(&self, tensor_id: usize) -> BrainResult<Tensor> {
        let mut guard = self.storage.lock().unwrap();
        guard.remove(&tensor_id).ok_or_else(|| {
            BrainError::invalid_value(format!(
                "Offloaded tensor {} not found in host storage",
                tensor_id
            ))
        })
    }

    /// Clears all host storage.
    pub fn clear(&self) {
        self.storage.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
