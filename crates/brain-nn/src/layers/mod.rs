//! # Neural Network Layers
//!
//! Linear, Convolution, Attention, Normalization, Pooling, Recurrent, and Embedding layers.
#![allow(missing_docs)]

pub mod linear;
pub mod linear2d;
pub mod conv;
pub mod conv2d;
pub mod conv_transpose;
pub mod attention;
pub mod multihead;
pub mod norm;
pub mod pool;
pub mod recurrent;
pub mod rnn_cells;
pub mod embedding;
pub mod activation_layers;
pub mod pixel_shuffle;

pub use linear::Linear;
pub use linear2d::{Bilinear, Identity};
pub use conv::Conv2d;
pub use conv_transpose::ConvTranspose2d;
pub use attention::{MultiheadAttention, scaled_dot_product_attention, AttentionConfig};
pub use norm::LayerNorm;
pub use pool::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d};
pub use pixel_shuffle::PixelShuffle;
pub use recurrent::{LSTM, GRU};
pub use embedding::Embedding;

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
