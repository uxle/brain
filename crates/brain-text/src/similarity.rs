//! # String & Vector Similarity Metrics
//!
//! Edit distances, token set similarities, Jaro-Winkler, cosine, and pairwise similarity matrices.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::utils::{cosine_similarity_slice, damerau_levenshtein_distance, jaccard_similarity, levenshtein_distance};

/// Categorical similarity algorithm metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SimilarityMetric {
    /// Cosine vector similarity.
    #[default]
    Cosine,
    /// Dot product.
    DotProduct,
    /// Euclidean distance similarity: $1 / (1 + d)$.
    Euclidean,
    /// Manhattan distance similarity: $1 / (1 + d)$.
    Manhattan,
    /// Jaccard token set similarity.
    Jaccard,
    /// Normalized Levenshtein similarity: $1 - d / \max(|s_1|, |s_2|)$.
    Levenshtein,
    /// Normalized Damerau-Levenshtein similarity.
    DamerauLevenshtein,
    /// Jaro similarity metric.
    Jaro,
    /// Jaro-Winkler metric with prefix scaling.
    JaroWinkler,
    /// Sørensen-Dice coefficient: $2 |A \cap B| / (|A| + |B|)$.
    SorensenDice,
}

/// Configuration for similarity computation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SimilarityConfig {
    /// Selected similarity metric.
    pub metric: SimilarityMetric,
    /// Whether character comparison is case sensitive.
    pub case_sensitive: bool,
    /// Normalize outputs to range [0.0, 1.0].
    pub normalize: bool,
}

/// Unified API for text and vector similarity metrics.
pub struct TextSimilarity;

impl TextSimilarity {
    /// Computes cosine similarity between two float slices.
    pub fn cosine(v1: &[f32], v2: &[f32]) -> f32 {
        cosine_similarity_slice(v1, v2)
    }

    /// Computes dot product between two float slices.
    pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return 0.0;
        }
        v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum()
    }

    /// Computes Euclidean distance similarity: $1 / (1 + \sqrt{\sum (a_i - b_i)^2})$.
    pub fn euclidean(v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return 0.0;
        }
        let dist_sq: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
        1.0 / (1.0 + dist_sq.sqrt())
    }

    /// Computes Manhattan distance similarity: $1 / (1 + \sum |a_i - b_i|)$.
    pub fn manhattan(v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() {
            return 0.0;
        }
        let dist: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| (a - b).abs()).sum();
        1.0 / (1.0 + dist)
    }

    /// Computes normalized Levenshtein similarity in `[0.0, 1.0]`.
    pub fn levenshtein_similarity(s1: &str, s2: &str) -> f64 {
        let max_len = s1.chars().count().max(s2.chars().count());
        if max_len == 0 {
            return 1.0;
        }
        let dist = levenshtein_distance(s1, s2);
        1.0 - (dist as f64 / max_len as f64)
    }

    /// Computes normalized Damerau-Levenshtein similarity in `[0.0, 1.0]`.
    pub fn damerau_similarity(s1: &str, s2: &str) -> f64 {
        let max_len = s1.chars().count().max(s2.chars().count());
        if max_len == 0 {
            return 1.0;
        }
        let dist = damerau_levenshtein_distance(s1, s2);
        1.0 - (dist as f64 / max_len as f64)
    }

    /// Computes Jaro similarity metric between two strings.
    pub fn jaro(s1: &str, s2: &str) -> f64 {
        let a: Vec<char> = s1.chars().collect();
        let b: Vec<char> = s2.chars().collect();
        let len_a = a.len();
        let len_b = b.len();

        if len_a == 0 && len_b == 0 {
            return 1.0;
        }
        if len_a == 0 || len_b == 0 {
            return 0.0;
        }

        let match_distance = (len_a.max(len_b) / 2).saturating_sub(1);
        let mut a_matches = vec![false; len_a];
        let mut b_matches = vec![false; len_b];
        let mut matches = 0usize;

        for i in 0..len_a {
            let start = i.saturating_sub(match_distance);
            let end = (i + match_distance + 1).min(len_b);

            for j in start..end {
                if !b_matches[j] && a[i] == b[j] {
                    a_matches[i] = true;
                    b_matches[j] = true;
                    matches += 1;
                    break;
                }
            }
        }

        if matches == 0 {
            return 0.0;
        }

        let mut transpositions = 0usize;
        let mut b_idx = 0usize;

        for i in 0..len_a {
            if a_matches[i] {
                while !b_matches[b_idx] {
                    b_idx += 1;
                }
                if a[i] != b[b_idx] {
                    transpositions += 1;
                }
                b_idx += 1;
            }
        }

        let m = matches as f64;
        let t = (transpositions / 2) as f64;
        ((m / len_a as f64) + (m / len_b as f64) + ((m - t) / m)) / 3.0
    }

    /// Computes Jaro-Winkler similarity with common prefix bonus.
    pub fn jaro_winkler(s1: &str, s2: &str) -> f64 {
        let jaro_dist = Self::jaro(s1, s2);
        let a: Vec<char> = s1.chars().collect();
        let b: Vec<char> = s2.chars().collect();

        let mut prefix_len = 0usize;
        let max_prefix = 4.min(a.len()).min(b.len());

        for i in 0..max_prefix {
            if a[i] == b[i] {
                prefix_len += 1;
            } else {
                break;
            }
        }

        let p = 0.1;
        jaro_dist + (prefix_len as f64 * p * (1.0 - jaro_dist))
    }

    /// Computes Jaccard set similarity between token slices.
    pub fn jaccard(s1: &[String], s2: &[String]) -> f64 {
        jaccard_similarity(s1, s2)
    }

    /// Computes Sørensen-Dice coefficient: $2 |A \cap B| / (|A| + |B|)$.
    pub fn sorensen_dice(s1: &[String], s2: &[String]) -> f64 {
        if s1.is_empty() && s2.is_empty() {
            return 1.0;
        }
        let set1: std::collections::HashSet<&String> = s1.iter().collect();
        let set2: std::collections::HashSet<&String> = s2.iter().collect();
        let intersection = set1.intersection(&set2).count();
        (2.0 * intersection as f64) / (s1.len() + s2.len()) as f64
    }

    /// Computes character n-gram overlap between two strings.
    pub fn n_gram_overlap(s1: &str, s2: &str, n: usize) -> f64 {
        let g1 = crate::text_ops::shingles(s1, n);
        let g2 = crate::text_ops::shingles(s2, n);
        if g1.is_empty() && g2.is_empty() {
            return 1.0;
        }
        let inter = g1.intersection(&g2).count();
        let union = g1.union(&g2).count();
        if union == 0 {
            1.0
        } else {
            inter as f64 / union as f64
        }
    }

    /// Generates full pairwise cosine similarity matrix for a list of vectors.
    pub fn pairwise_similarity_matrix(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = vectors.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let sim = Self::cosine(&vectors[i], &vectors[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision, clippy::float_cmp, clippy::len_zero)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::vocab::*;
    use crate::text_ops::*;
    use crate::features::*;
    use crate::similarity::*;
    use crate::lm::*;
    use crate::process::*;
    use crate::optimize::*;
    use crate::analyze::*;
    use crate::compute::*;
    use crate::helper::*;
    use crate::transform::*;
    use crate::builder::*;
    use crate::tokenizer::*;
    use crate::tokenizer::bpe::*;
    use crate::tokenizer::sentencepiece::*;
    use crate::tokenizer::wordpiece::*;
    use crate::tokenizer::char::*;
    use crate::tokenizer::trainer::*;
    use crate::tokenizer::normalizer::*;
    use crate::tokenizer::pretokenizer::*;
    use crate::tokenizer::bytelevel::*;
    use crate::tokenizer::post::*;
    use crate::embedding::*;
    use crate::embedding::pretrained::*;
    use crate::embedding::fasttext::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_similarity_metrics_1() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!((TextSimilarity::cosine(&v1, &v2) - 1.0).abs() < 1e-5);
        assert!((TextSimilarity::euclidean(&v1, &v2) - 1.0).abs() < 1e-5);
        assert_eq!(TextSimilarity::dot_product(&v1, &v2), 14.0);

        let lev = TextSimilarity::levenshtein_similarity("martha", "marhta");
        assert!(lev > 0.6);

        let dam = TextSimilarity::damerau_similarity("martha", "marhta");
        assert!(dam >= lev);

        let jaro = TextSimilarity::jaro("DWAYNE", "DUANE");
        assert!(jaro > 0.8);

        let jw = TextSimilarity::jaro_winkler("DWAYNE", "DUANE");
        assert!(jw >= jaro);

        let t1 = vec!["neural".to_string(), "network_1".to_string()];
        let t2 = vec!["neural".to_string(), "deep_1".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }
}
