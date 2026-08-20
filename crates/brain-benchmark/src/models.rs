//! # Full Model Benchmark Suites
//!
//! Complete workload benchmarks for standard deep learning architectures:
//! Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Transformer Encoder blocks.

use crate::core::{BenchConfig, BenchResult};
use crate::runner::Runner;
use brain_core::tensor::arithmetic as arith_t;
use brain_core::tensor::math as math_t;
use brain_core::{BrainResult, Tensor};

/// Configuration options for synthetic model benchmarks.
#[derive(Debug, Clone)]
pub struct ModelBenchConfig {
    pub batch_size: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
}

impl Default for ModelBenchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            input_dim: 128,
            hidden_dim: 256,
            num_layers: 4,
        }
    }
}

/// Benchmarks a Multi-Layer Perceptron (MLP) forward pass.
pub fn bench_mlp(config: &ModelBenchConfig) -> BrainResult<BenchResult> {
    let x = Tensor::ones(vec![config.batch_size, config.input_dim]);
    let w1 = Tensor::ones(vec![config.input_dim, config.hidden_dim]);
    let w2 = Tensor::ones(vec![config.hidden_dim, config.hidden_dim]);

    let bench_cfg = BenchConfig::new(format!(
        "mlp_b{}_h{}_l{}",
        config.batch_size, config.hidden_dim, config.num_layers
    ))
    .with_tag("model")
    .with_tag("mlp");

    Runner::run_benchmark(&bench_cfg, || {
        let mut h = math_t::relu(&arith_t::matmul(&x, &w1));
        for _ in 1..config.num_layers {
            h = math_t::relu(&arith_t::matmul(&h, &w2));
        }
        std::hint::black_box(h);
    })
}

/// Benchmarks a Transformer Encoder forward block: `Q @ K^T -> Softmax -> @ V -> Linear -> Add`.
pub fn bench_transformer_layer(
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
) -> BrainResult<BenchResult> {
    let q = Tensor::ones(vec![batch_size * seq_len, d_model]);
    let k = Tensor::ones(vec![batch_size * seq_len, d_model]);
    let v = Tensor::ones(vec![batch_size * seq_len, d_model]);

    let bench_cfg = BenchConfig::new(format!(
        "transformer_b{}_s{}_d{}",
        batch_size, seq_len, d_model
    ))
    .with_tag("model")
    .with_tag("transformer");

    Runner::run_benchmark(&bench_cfg, || {
        let kt = k.transpose(0, 1);
        let scores = arith_t::matmul(&q, &kt);
        let attn_weights = brain_core::tensor::special::softmax(&scores, 1);
        let out = arith_t::matmul(&attn_weights, &v);
        std::hint::black_box(out);
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
