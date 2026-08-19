//! # WebNN MLGraphBuilder Op Mappings
//!
//! WebNN standard operator signatures and attribute schemas.

/// Maps standard op identifier to WebNN `MLGraphBuilder` method name.
pub fn map_to_webnn_op(op_name: &str) -> Option<&'static str> {
    match op_name {
        "Add" => Some("add"),
        "Sub" => Some("sub"),
        "Mul" => Some("mul"),
        "Div" => Some("div"),
        "MatMul" => Some("matmul"),
        "Conv2d" => Some("conv2d"),
        "Relu" => Some("relu"),
        "Softmax" => Some("softmax"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
