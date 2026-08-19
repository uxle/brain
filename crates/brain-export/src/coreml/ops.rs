//! # CoreML Neural Network Layer Mappings
//!
//! Layer definitions and activation parameters for CoreML neural network proto schemas.

/// Checks if op is directly convertible to a native CoreML neural network layer.
pub fn is_coreml_layer_supported(op_name: &str) -> bool {
    matches!(
        op_name,
        "Convolution" | "InnerProduct" | "Activation" | "Pooling" | "Softmax"
    )
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
