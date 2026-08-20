//! # Key-Value (KV) Cache for High-Throughput Autoregressive Inference
//!
//! Multi-layer cached state management preventing redundant key/value recomputation during sequential token generation.
#![allow(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unused_mut,
    unused_comparisons,
    clippy::all
)]

use crate::core::{TransformerError, TransformerResult};
use brain_core::Tensor;

/// Configuration for Key-Value Cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Head dimension $d_k$.
    pub head_dim: usize,
    /// Maximum pre-allocated sequence capacity.
    pub max_seq_len: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 12,
            batch_size: 1,
            num_kv_heads: 12,
            head_dim: 64,
            max_seq_len: 2048,
        }
    }
}

/// Single layer Key and Value cache buffer.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    /// Cached keys tensor `[batch_size, num_kv_heads, current_len, head_dim]`.
    pub key_cache: Vec<f64>,
    /// Cached values tensor `[batch_size, num_kv_heads, current_len, head_dim]`.
    pub value_cache: Vec<f64>,
    /// Current cached sequence length.
    pub current_seq_len: usize,
    /// Maximum context capacity.
    pub max_seq_len: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl LayerKvCache {
    /// Creates a new `LayerKvCache` with pre-allocated memory.
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            key_cache: Vec::with_capacity(batch_size * num_kv_heads * max_seq_len * head_dim),
            value_cache: Vec::with_capacity(batch_size * num_kv_heads * max_seq_len * head_dim),
            current_seq_len: 0,
            max_seq_len,
            batch_size,
            num_kv_heads,
            head_dim,
        }
    }

    /// Appends new key and value tensors `[batch_size, num_kv_heads, new_seq_len, head_dim]` and returns full cached 4D tensors.
    pub fn update(
        &mut self,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> TransformerResult<(Tensor, Tensor)> {
        let shape_k = new_k.shape();
        let new_len = shape_k[2];

        if self.current_seq_len + new_len > self.max_seq_len {
            return Err(TransformerError::CacheError(format!(
                "Sequence length {} exceeds cache capacity {}",
                self.current_seq_len + new_len,
                self.max_seq_len
            )));
        }

        let k_data = new_k.data();
        let v_data = new_v.data();

        // If starting fresh or appending sequentially
        if self.current_seq_len == 0 {
            self.key_cache.extend_from_slice(k_data);
            self.value_cache.extend_from_slice(v_data);
        } else {
            // Interleave append across batch and heads
            let total_len = self.current_seq_len + new_len;
            let mut updated_k =
                vec![0.0f64; self.batch_size * self.num_kv_heads * total_len * self.head_dim];
            let mut updated_v =
                vec![0.0f64; self.batch_size * self.num_kv_heads * total_len * self.head_dim];

            for b in 0..self.batch_size {
                for h in 0..self.num_kv_heads {
                    let old_offset =
                        (b * self.num_kv_heads + h) * self.current_seq_len * self.head_dim;
                    let new_offset = (b * self.num_kv_heads + h) * new_len * self.head_dim;
                    let out_offset = (b * self.num_kv_heads + h) * total_len * self.head_dim;

                    // Copy old cache
                    updated_k[out_offset..out_offset + self.current_seq_len * self.head_dim]
                        .copy_from_slice(
                            &self.key_cache
                                [old_offset..old_offset + self.current_seq_len * self.head_dim],
                        );
                    updated_v[out_offset..out_offset + self.current_seq_len * self.head_dim]
                        .copy_from_slice(
                            &self.value_cache
                                [old_offset..old_offset + self.current_seq_len * self.head_dim],
                        );

                    // Append new chunk
                    let append_out_offset = out_offset + self.current_seq_len * self.head_dim;
                    updated_k[append_out_offset..append_out_offset + new_len * self.head_dim]
                        .copy_from_slice(&k_data[new_offset..new_offset + new_len * self.head_dim]);
                    updated_v[append_out_offset..append_out_offset + new_len * self.head_dim]
                        .copy_from_slice(&v_data[new_offset..new_offset + new_len * self.head_dim]);
                }
            }

            self.key_cache = updated_k;
            self.value_cache = updated_v;
        }

        self.current_seq_len += new_len;

        let total_shape = vec![
            self.batch_size,
            self.num_kv_heads,
            self.current_seq_len,
            self.head_dim,
        ];
        let full_k = Tensor::from_vec(self.key_cache.clone(), total_shape.clone());
        let full_v = Tensor::from_vec(self.value_cache.clone(), total_shape);

        Ok((full_k, full_v))
    }

    /// Resets cached sequences to empty.
    pub fn clear(&mut self) {
        self.key_cache.clear();
        self.value_cache.clear();
        self.current_seq_len = 0;
    }
}

/// Multi-Layer KV Cache container.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Layer-wise caches.
    pub layers: Vec<LayerKvCache>,
    /// Configuration options.
    pub config: KvCacheConfig,
}

impl KvCache {
    /// Creates a new multi-layer `KvCache`.
    pub fn new(config: KvCacheConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LayerKvCache::new(
                config.batch_size,
                config.num_kv_heads,
                config.head_dim,
                config.max_seq_len,
            ));
        }
        Self { layers, config }
    }

    /// Updates cache for layer `layer_idx`.
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> TransformerResult<(Tensor, Tensor)> {
        if layer_idx >= self.layers.len() {
            return Err(TransformerError::CacheError(format!(
                "Layer index {} exceeds cache layer count {}",
                layer_idx,
                self.layers.len()
            )));
        }
        self.layers[layer_idx].update(new_k, new_v)
    }

    /// Clears all layer caches.
    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }

    /// Returns current sequence length of first layer.
    pub fn current_seq_len(&self) -> usize {
        self.layers.first().map(|l| l.current_seq_len).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision,
        clippy::float_cmp,
        clippy::len_zero,
        clippy::all
    )]
    use super::*;
    use crate::attention::flash_lite::*;
    use crate::attention::multi_head::*;
    use crate::attention::multi_query::*;
    use crate::attention::relative::*;
    use crate::attention::scaled::*;
    use crate::attention::xformers_lite::*;
    use crate::attention::*;
    use crate::builder::*;
    use crate::config::*;
    use crate::core::*;
    use crate::decoder::cross::*;
    use crate::decoder::layer::*;
    use crate::decoder::*;
    use crate::embedding_layers::*;
    use crate::encoder::block::*;
    use crate::encoder::layer::*;
    use crate::encoder::*;
    use crate::ffn::*;
    use crate::generate::*;
    use crate::head::*;
    use crate::kv_cache::*;
    use crate::models::bert_lite::*;
    use crate::models::gpt_lite::*;
    use crate::models::llama_lite::*;
    use crate::models::t5_lite::*;
    use crate::models::*;
    use crate::ops::*;
    use crate::position::alibi::*;
    use crate::position::learned::*;
    use crate::position::rope::*;
    use crate::position::*;
    use crate::utils::*;
    use brain_core::Tensor;

    #[test]
    fn test_kv_cache_pipeline_1() {
        let cfg = KvCacheConfig {
            num_layers: 2,
            batch_size: 1,
            num_kv_heads: 2,
            head_dim: 8,
            max_seq_len: 32,
        };
        let mut cache = KvCache::new(cfg);
        assert_eq!(cache.current_seq_len(), 0);

        let k1 = Tensor::from_vec(vec![1.0; 1 * 2 * 3 * 8], vec![1, 2, 3, 8]);
        let v1 = Tensor::from_vec(vec![2.0; 1 * 2 * 3 * 8], vec![1, 2, 3, 8]);

        let (full_k, full_v) = cache.update(0, &k1, &v1).unwrap();
        assert_eq!(full_k.shape(), &[1, 2, 3, 8]);
        assert_eq!(full_v.shape(), &[1, 2, 3, 8]);
        assert_eq!(cache.current_seq_len(), 3);

        let k2 = Tensor::from_vec(vec![3.0; 1 * 2 * 1 * 8], vec![1, 2, 1, 8]);
        let v2 = Tensor::from_vec(vec![4.0; 1 * 2 * 1 * 8], vec![1, 2, 1, 8]);

        let (full_k2, _) = cache.update(0, &k2, &v2).unwrap();
        assert_eq!(full_k2.shape(), &[1, 2, 4, 8]);
        assert_eq!(cache.current_seq_len(), 4);

        cache.clear();
        assert_eq!(cache.current_seq_len(), 0);
    }
}
