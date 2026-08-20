//! # Neural Network Layers
//!
//! Linear, Convolution, Attention, Normalization, Pooling, Recurrent, and Embedding layers.
#![allow(missing_docs)]

pub mod activation_layers;
pub mod attention;
pub mod conv;
pub mod conv1d;
pub mod conv2d;
pub mod conv_transpose;
pub mod embedding;
pub mod linear;
pub mod linear2d;
pub mod multihead;
pub mod norm;
pub mod pixel_shuffle;
pub mod pool;
pub mod recurrent;
pub mod rnn_cells;

pub use attention::{scaled_dot_product_attention, AttentionConfig, MultiheadAttention};
pub use conv::Conv2d;
pub use conv1d::Conv1d;
pub use conv_transpose::ConvTranspose2d;
pub use embedding::Embedding;
pub use linear::Linear;
pub use linear2d::{Bilinear, Identity};
pub use norm::LayerNorm;
pub use pixel_shuffle::PixelShuffle;
pub use pool::{AdaptiveAvgPool2d, AdaptiveMaxPool2d, AvgPool2d, MaxPool2d};
pub use recurrent::{GRU, LSTM};

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
