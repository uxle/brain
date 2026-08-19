//! # Conditioning Signals & Cross-Attention Injection
//!
//! Text embeddings, class labels, and image inpainting conditioning masks.

use brain_core::Tensor;

/// Multimodal conditioning context.
pub struct ConditioningContext {
    pub text_emb: Option<Tensor>,
    pub class_label: Option<usize>,
}

impl Default for ConditioningContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ConditioningContext {
    /// Creates an empty `ConditioningContext`.
    pub fn new() -> Self {
        Self {
            text_emb: None,
            class_label: None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
