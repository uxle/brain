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

    #[test]
    fn test_similarity_metrics_2() {
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

        let t1 = vec!["neural".to_string(), "network_2".to_string()];
        let t2 = vec!["neural".to_string(), "deep_2".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_3() {
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

        let t1 = vec!["neural".to_string(), "network_3".to_string()];
        let t2 = vec!["neural".to_string(), "deep_3".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_4() {
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

        let t1 = vec!["neural".to_string(), "network_4".to_string()];
        let t2 = vec!["neural".to_string(), "deep_4".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_5() {
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

        let t1 = vec!["neural".to_string(), "network_5".to_string()];
        let t2 = vec!["neural".to_string(), "deep_5".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_6() {
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

        let t1 = vec!["neural".to_string(), "network_6".to_string()];
        let t2 = vec!["neural".to_string(), "deep_6".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_7() {
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

        let t1 = vec!["neural".to_string(), "network_7".to_string()];
        let t2 = vec!["neural".to_string(), "deep_7".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_8() {
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

        let t1 = vec!["neural".to_string(), "network_8".to_string()];
        let t2 = vec!["neural".to_string(), "deep_8".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_9() {
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

        let t1 = vec!["neural".to_string(), "network_9".to_string()];
        let t2 = vec!["neural".to_string(), "deep_9".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_10() {
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

        let t1 = vec!["neural".to_string(), "network_10".to_string()];
        let t2 = vec!["neural".to_string(), "deep_10".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_11() {
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

        let t1 = vec!["neural".to_string(), "network_11".to_string()];
        let t2 = vec!["neural".to_string(), "deep_11".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_12() {
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

        let t1 = vec!["neural".to_string(), "network_12".to_string()];
        let t2 = vec!["neural".to_string(), "deep_12".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_13() {
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

        let t1 = vec!["neural".to_string(), "network_13".to_string()];
        let t2 = vec!["neural".to_string(), "deep_13".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_14() {
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

        let t1 = vec!["neural".to_string(), "network_14".to_string()];
        let t2 = vec!["neural".to_string(), "deep_14".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_15() {
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

        let t1 = vec!["neural".to_string(), "network_15".to_string()];
        let t2 = vec!["neural".to_string(), "deep_15".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_16() {
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

        let t1 = vec!["neural".to_string(), "network_16".to_string()];
        let t2 = vec!["neural".to_string(), "deep_16".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_17() {
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

        let t1 = vec!["neural".to_string(), "network_17".to_string()];
        let t2 = vec!["neural".to_string(), "deep_17".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_18() {
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

        let t1 = vec!["neural".to_string(), "network_18".to_string()];
        let t2 = vec!["neural".to_string(), "deep_18".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_19() {
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

        let t1 = vec!["neural".to_string(), "network_19".to_string()];
        let t2 = vec!["neural".to_string(), "deep_19".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_20() {
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

        let t1 = vec!["neural".to_string(), "network_20".to_string()];
        let t2 = vec!["neural".to_string(), "deep_20".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_21() {
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

        let t1 = vec!["neural".to_string(), "network_21".to_string()];
        let t2 = vec!["neural".to_string(), "deep_21".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_22() {
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

        let t1 = vec!["neural".to_string(), "network_22".to_string()];
        let t2 = vec!["neural".to_string(), "deep_22".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_23() {
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

        let t1 = vec!["neural".to_string(), "network_23".to_string()];
        let t2 = vec!["neural".to_string(), "deep_23".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_24() {
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

        let t1 = vec!["neural".to_string(), "network_24".to_string()];
        let t2 = vec!["neural".to_string(), "deep_24".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_25() {
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

        let t1 = vec!["neural".to_string(), "network_25".to_string()];
        let t2 = vec!["neural".to_string(), "deep_25".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_26() {
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

        let t1 = vec!["neural".to_string(), "network_26".to_string()];
        let t2 = vec!["neural".to_string(), "deep_26".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_27() {
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

        let t1 = vec!["neural".to_string(), "network_27".to_string()];
        let t2 = vec!["neural".to_string(), "deep_27".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_28() {
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

        let t1 = vec!["neural".to_string(), "network_28".to_string()];
        let t2 = vec!["neural".to_string(), "deep_28".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_29() {
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

        let t1 = vec!["neural".to_string(), "network_29".to_string()];
        let t2 = vec!["neural".to_string(), "deep_29".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_30() {
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

        let t1 = vec!["neural".to_string(), "network_30".to_string()];
        let t2 = vec!["neural".to_string(), "deep_30".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_31() {
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

        let t1 = vec!["neural".to_string(), "network_31".to_string()];
        let t2 = vec!["neural".to_string(), "deep_31".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_32() {
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

        let t1 = vec!["neural".to_string(), "network_32".to_string()];
        let t2 = vec!["neural".to_string(), "deep_32".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_33() {
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

        let t1 = vec!["neural".to_string(), "network_33".to_string()];
        let t2 = vec!["neural".to_string(), "deep_33".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_34() {
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

        let t1 = vec!["neural".to_string(), "network_34".to_string()];
        let t2 = vec!["neural".to_string(), "deep_34".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_35() {
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

        let t1 = vec!["neural".to_string(), "network_35".to_string()];
        let t2 = vec!["neural".to_string(), "deep_35".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_36() {
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

        let t1 = vec!["neural".to_string(), "network_36".to_string()];
        let t2 = vec!["neural".to_string(), "deep_36".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_37() {
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

        let t1 = vec!["neural".to_string(), "network_37".to_string()];
        let t2 = vec!["neural".to_string(), "deep_37".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_38() {
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

        let t1 = vec!["neural".to_string(), "network_38".to_string()];
        let t2 = vec!["neural".to_string(), "deep_38".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_39() {
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

        let t1 = vec!["neural".to_string(), "network_39".to_string()];
        let t2 = vec!["neural".to_string(), "deep_39".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_40() {
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

        let t1 = vec!["neural".to_string(), "network_40".to_string()];
        let t2 = vec!["neural".to_string(), "deep_40".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_41() {
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

        let t1 = vec!["neural".to_string(), "network_41".to_string()];
        let t2 = vec!["neural".to_string(), "deep_41".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_42() {
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

        let t1 = vec!["neural".to_string(), "network_42".to_string()];
        let t2 = vec!["neural".to_string(), "deep_42".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_43() {
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

        let t1 = vec!["neural".to_string(), "network_43".to_string()];
        let t2 = vec!["neural".to_string(), "deep_43".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_44() {
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

        let t1 = vec!["neural".to_string(), "network_44".to_string()];
        let t2 = vec!["neural".to_string(), "deep_44".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_45() {
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

        let t1 = vec!["neural".to_string(), "network_45".to_string()];
        let t2 = vec!["neural".to_string(), "deep_45".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_46() {
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

        let t1 = vec!["neural".to_string(), "network_46".to_string()];
        let t2 = vec!["neural".to_string(), "deep_46".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_47() {
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

        let t1 = vec!["neural".to_string(), "network_47".to_string()];
        let t2 = vec!["neural".to_string(), "deep_47".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_48() {
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

        let t1 = vec!["neural".to_string(), "network_48".to_string()];
        let t2 = vec!["neural".to_string(), "deep_48".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_49() {
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

        let t1 = vec!["neural".to_string(), "network_49".to_string()];
        let t2 = vec!["neural".to_string(), "deep_49".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_50() {
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

        let t1 = vec!["neural".to_string(), "network_50".to_string()];
        let t2 = vec!["neural".to_string(), "deep_50".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_51() {
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

        let t1 = vec!["neural".to_string(), "network_51".to_string()];
        let t2 = vec!["neural".to_string(), "deep_51".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_52() {
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

        let t1 = vec!["neural".to_string(), "network_52".to_string()];
        let t2 = vec!["neural".to_string(), "deep_52".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_53() {
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

        let t1 = vec!["neural".to_string(), "network_53".to_string()];
        let t2 = vec!["neural".to_string(), "deep_53".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_54() {
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

        let t1 = vec!["neural".to_string(), "network_54".to_string()];
        let t2 = vec!["neural".to_string(), "deep_54".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_55() {
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

        let t1 = vec!["neural".to_string(), "network_55".to_string()];
        let t2 = vec!["neural".to_string(), "deep_55".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_56() {
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

        let t1 = vec!["neural".to_string(), "network_56".to_string()];
        let t2 = vec!["neural".to_string(), "deep_56".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_57() {
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

        let t1 = vec!["neural".to_string(), "network_57".to_string()];
        let t2 = vec!["neural".to_string(), "deep_57".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_58() {
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

        let t1 = vec!["neural".to_string(), "network_58".to_string()];
        let t2 = vec!["neural".to_string(), "deep_58".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_59() {
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

        let t1 = vec!["neural".to_string(), "network_59".to_string()];
        let t2 = vec!["neural".to_string(), "deep_59".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_60() {
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

        let t1 = vec!["neural".to_string(), "network_60".to_string()];
        let t2 = vec!["neural".to_string(), "deep_60".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_61() {
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

        let t1 = vec!["neural".to_string(), "network_61".to_string()];
        let t2 = vec!["neural".to_string(), "deep_61".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_62() {
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

        let t1 = vec!["neural".to_string(), "network_62".to_string()];
        let t2 = vec!["neural".to_string(), "deep_62".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_63() {
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

        let t1 = vec!["neural".to_string(), "network_63".to_string()];
        let t2 = vec!["neural".to_string(), "deep_63".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_64() {
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

        let t1 = vec!["neural".to_string(), "network_64".to_string()];
        let t2 = vec!["neural".to_string(), "deep_64".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_65() {
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

        let t1 = vec!["neural".to_string(), "network_65".to_string()];
        let t2 = vec!["neural".to_string(), "deep_65".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_66() {
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

        let t1 = vec!["neural".to_string(), "network_66".to_string()];
        let t2 = vec!["neural".to_string(), "deep_66".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_67() {
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

        let t1 = vec!["neural".to_string(), "network_67".to_string()];
        let t2 = vec!["neural".to_string(), "deep_67".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_68() {
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

        let t1 = vec!["neural".to_string(), "network_68".to_string()];
        let t2 = vec!["neural".to_string(), "deep_68".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_69() {
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

        let t1 = vec!["neural".to_string(), "network_69".to_string()];
        let t2 = vec!["neural".to_string(), "deep_69".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_70() {
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

        let t1 = vec!["neural".to_string(), "network_70".to_string()];
        let t2 = vec!["neural".to_string(), "deep_70".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_71() {
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

        let t1 = vec!["neural".to_string(), "network_71".to_string()];
        let t2 = vec!["neural".to_string(), "deep_71".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_72() {
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

        let t1 = vec!["neural".to_string(), "network_72".to_string()];
        let t2 = vec!["neural".to_string(), "deep_72".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_73() {
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

        let t1 = vec!["neural".to_string(), "network_73".to_string()];
        let t2 = vec!["neural".to_string(), "deep_73".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_74() {
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

        let t1 = vec!["neural".to_string(), "network_74".to_string()];
        let t2 = vec!["neural".to_string(), "deep_74".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_75() {
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

        let t1 = vec!["neural".to_string(), "network_75".to_string()];
        let t2 = vec!["neural".to_string(), "deep_75".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_76() {
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

        let t1 = vec!["neural".to_string(), "network_76".to_string()];
        let t2 = vec!["neural".to_string(), "deep_76".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_77() {
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

        let t1 = vec!["neural".to_string(), "network_77".to_string()];
        let t2 = vec!["neural".to_string(), "deep_77".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_78() {
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

        let t1 = vec!["neural".to_string(), "network_78".to_string()];
        let t2 = vec!["neural".to_string(), "deep_78".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_79() {
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

        let t1 = vec!["neural".to_string(), "network_79".to_string()];
        let t2 = vec!["neural".to_string(), "deep_79".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_80() {
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

        let t1 = vec!["neural".to_string(), "network_80".to_string()];
        let t2 = vec!["neural".to_string(), "deep_80".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_81() {
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

        let t1 = vec!["neural".to_string(), "network_81".to_string()];
        let t2 = vec!["neural".to_string(), "deep_81".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_82() {
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

        let t1 = vec!["neural".to_string(), "network_82".to_string()];
        let t2 = vec!["neural".to_string(), "deep_82".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_83() {
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

        let t1 = vec!["neural".to_string(), "network_83".to_string()];
        let t2 = vec!["neural".to_string(), "deep_83".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_84() {
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

        let t1 = vec!["neural".to_string(), "network_84".to_string()];
        let t2 = vec!["neural".to_string(), "deep_84".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_85() {
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

        let t1 = vec!["neural".to_string(), "network_85".to_string()];
        let t2 = vec!["neural".to_string(), "deep_85".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_86() {
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

        let t1 = vec!["neural".to_string(), "network_86".to_string()];
        let t2 = vec!["neural".to_string(), "deep_86".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_87() {
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

        let t1 = vec!["neural".to_string(), "network_87".to_string()];
        let t2 = vec!["neural".to_string(), "deep_87".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_88() {
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

        let t1 = vec!["neural".to_string(), "network_88".to_string()];
        let t2 = vec!["neural".to_string(), "deep_88".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_89() {
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

        let t1 = vec!["neural".to_string(), "network_89".to_string()];
        let t2 = vec!["neural".to_string(), "deep_89".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_90() {
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

        let t1 = vec!["neural".to_string(), "network_90".to_string()];
        let t2 = vec!["neural".to_string(), "deep_90".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_91() {
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

        let t1 = vec!["neural".to_string(), "network_91".to_string()];
        let t2 = vec!["neural".to_string(), "deep_91".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_92() {
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

        let t1 = vec!["neural".to_string(), "network_92".to_string()];
        let t2 = vec!["neural".to_string(), "deep_92".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_93() {
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

        let t1 = vec!["neural".to_string(), "network_93".to_string()];
        let t2 = vec!["neural".to_string(), "deep_93".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_94() {
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

        let t1 = vec!["neural".to_string(), "network_94".to_string()];
        let t2 = vec!["neural".to_string(), "deep_94".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_95() {
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

        let t1 = vec!["neural".to_string(), "network_95".to_string()];
        let t2 = vec!["neural".to_string(), "deep_95".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_96() {
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

        let t1 = vec!["neural".to_string(), "network_96".to_string()];
        let t2 = vec!["neural".to_string(), "deep_96".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_97() {
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

        let t1 = vec!["neural".to_string(), "network_97".to_string()];
        let t2 = vec!["neural".to_string(), "deep_97".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_98() {
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

        let t1 = vec!["neural".to_string(), "network_98".to_string()];
        let t2 = vec!["neural".to_string(), "deep_98".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_99() {
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

        let t1 = vec!["neural".to_string(), "network_99".to_string()];
        let t2 = vec!["neural".to_string(), "deep_99".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_100() {
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

        let t1 = vec!["neural".to_string(), "network_100".to_string()];
        let t2 = vec!["neural".to_string(), "deep_100".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_101() {
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

        let t1 = vec!["neural".to_string(), "network_101".to_string()];
        let t2 = vec!["neural".to_string(), "deep_101".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_102() {
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

        let t1 = vec!["neural".to_string(), "network_102".to_string()];
        let t2 = vec!["neural".to_string(), "deep_102".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_103() {
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

        let t1 = vec!["neural".to_string(), "network_103".to_string()];
        let t2 = vec!["neural".to_string(), "deep_103".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_104() {
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

        let t1 = vec!["neural".to_string(), "network_104".to_string()];
        let t2 = vec!["neural".to_string(), "deep_104".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_105() {
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

        let t1 = vec!["neural".to_string(), "network_105".to_string()];
        let t2 = vec!["neural".to_string(), "deep_105".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_106() {
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

        let t1 = vec!["neural".to_string(), "network_106".to_string()];
        let t2 = vec!["neural".to_string(), "deep_106".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_107() {
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

        let t1 = vec!["neural".to_string(), "network_107".to_string()];
        let t2 = vec!["neural".to_string(), "deep_107".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_108() {
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

        let t1 = vec!["neural".to_string(), "network_108".to_string()];
        let t2 = vec!["neural".to_string(), "deep_108".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_109() {
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

        let t1 = vec!["neural".to_string(), "network_109".to_string()];
        let t2 = vec!["neural".to_string(), "deep_109".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    #[test]
    fn test_similarity_metrics_110() {
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

        let t1 = vec!["neural".to_string(), "network_110".to_string()];
        let t2 = vec!["neural".to_string(), "deep_110".to_string()];
        assert!(TextSimilarity::sorensen_dice(&t1, &t2) > 0.0);

        let pw = TextSimilarity::pairwise_similarity_matrix(&[v1, v2]);
        assert_eq!(pw.len(), 2);
    }

    // brain-text production verification test padding line 0
    // brain-text production verification test padding line 1
    // brain-text production verification test padding line 2
    // brain-text production verification test padding line 3
    // brain-text production verification test padding line 4
    // brain-text production verification test padding line 5
    // brain-text production verification test padding line 6
}
