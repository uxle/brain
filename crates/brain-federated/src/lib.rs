//! # brain-federated
//!
//! Production-grade federated learning with server/client architecture,
//! secure aggregation, differential privacy, and gradient compression.
//!
//! ## Modules
//! - [`client`] — local training loop and client reports
//! - [`server`] — round orchestration and weight aggregation (FedAvg/FedProx)
//! - [`privacy`] — Gaussian DP noise and secure aggregation masks
//! - [`compression`] — quantization and top-K sparsification
//! - [`monitor`] — convergence tracking and round history
//! - [`analyze`] — heterogeneity metrics and communication cost estimation
//! - [`builder`] — ergonomic system builder
//! - [`process`] — learning rate schedules and weight decay
//! - [`transform`] — Polyak averaging and weight normalization
//! - [`compute`] — gradient clipping and batched ops
//! - [`core`] — fundamental types: `ClientId`, `ModelDelta`, `ServerMetrics`
//! - [`config`] — `FedConfig` master configuration
//! - [`ops`] — tensor-level delta operations
//! - [`utils`] — client sampling and statistics helpers

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod analyze;
pub mod builder;
pub mod client;
pub mod compression;
pub mod compute;
pub mod config;
pub mod core;
pub mod impl_;
pub mod monitor;
pub mod ops;
pub mod privacy;
pub mod process;
pub mod server;
pub mod transform;
pub mod utils;

pub use analyze::{communication_cost_bytes, cosine_similarity_deltas, estimate_heterogeneity};
pub use builder::FedSystemBuilder;
pub use client::{ClientConfig, ClientReport, LocalTrainer};
pub use compression::{
    dequantize_tensor, quantize_tensor, top_k_sparsify, QuantConfig, SparseConfig,
};
pub use compute::{clip_grad_norm, global_grad_norm, multiply_accumulate};
pub use config::FedConfig;
pub use core::{ClientId, ClientMetrics, ModelDelta, RoundId, ServerMetrics};
pub use impl_::run_round;
pub use monitor::FedMonitor;
pub use ops::{l2_norm_delta, scale_delta};
pub use privacy::{add_dp_noise, mask_tensor, DpConfig, GaussianNoise, SecureAggregator};
pub use process::{apply_weight_decay, cosine_lr, mse_eval};
pub use server::{
    fed_avg_aggregate, median_aggregate, trimmed_mean_aggregate, AggregationAlgorithm,
    FederatedServer, RoundStats, ServerConfig,
};
pub use transform::{normalize_weights, polyak_average};
pub use utils::{sample_clients, stddev};
