//! # brain-nn
//!
//! Production-grade neural network layer library for the Brain Framework:
//! activations, modules, containers, initializations, normalization, dropout,
//! linear, convolution, multi-head attention, recurrent networks, embeddings, and pruning.
//!
//! ## Architecture
//! - [`module`] — `Module` trait, `Parameter`, `Buffer`, `NamedParameter`, `ModuleList`
//! - [`activations`] — ReLU, LeakyReLU, Sigmoid, Tanh, GELU, FastGELU, Softmax, LogSoftmax, SiLU/Swish, Mish
//! - [`init`] — Kaiming (He), Xavier (Glorot), Uniform, Normal, Orthogonal, and Residual schedules
//! - [`containers`] — `Sequential`, `SequentialNamed`, `ModuleList`
//! - [`layers`] — `Linear`, `Conv2d`, `ConvTranspose2d`, `MultiheadAttention`, `Embedding`, `LSTM`, `GRU`
//! - [`normalization`] — `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`
//! - [`dropout`] — `Dropout`, `AlphaDropout`, `Dropout2d`, `FusedDropout`
//! - [`hooks`] — `HookRegistry`, forward pre/post execution hooks
//! - [`pruning`] — `PruningMask`, magnitude and structured pruning

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod activations;
pub mod containers;
pub mod dropout;
pub mod hooks;
pub mod init;
pub mod layers;
pub mod module;
pub mod normalization;
pub mod pruning;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use activations::{
    fast_gelu, gelu, leaky_relu, log_softmax, mish, relu, sigmoid, silu, softmax, swish,
    tanh, Activation, ActivationKind, FastGELU, GELU, LeakyReLU, LogSoftmax, Mish, ReLU,
    Sigmoid, SiLU, Softmax, SoftmaxConfig, Swish, Tanh,
};
pub use containers::{NamedModule, Sequential, SequentialNamed};
pub use dropout::{AlphaDropout, Dropout, Dropout2d, FusedDropout};
pub use hooks::{ForwardPostHook, ForwardPreHook, HookRegistry};
pub use init::{
    calculate_fan, kaiming_normal, kaiming_uniform, normal_init, orthogonal_init,
    scaled_residual_init, uniform_init, xavier_normal, xavier_uniform, zero_init_last_layer,
    InitConfig, InitPolicy, InitScheme,
};
pub use layers::{
    scaled_dot_product_attention, AttentionConfig, AvgPool2d, Bilinear, Conv2d,
    ConvTranspose2d, Embedding, Identity, LayerNorm, Linear, MaxPool2d,
    MultiheadAttention, GRU, LSTM,
};
pub use layers::conv2d::ConvConfig;
pub use layers::multihead::MhaConfig;
pub use module::{Buffer, Module, ModuleError, ModuleList, ModuleResult, NamedParameter, Parameter};
pub use normalization::{
    BatchNorm2d, GroupNorm, NormalizationLayer, RMSNorm, RMSNormConfig,
};
pub use pruning::PruningMask;

/// Framework version string.
pub const VERSION: &str = "0.2.0";