//! # CLI Helper Utilities & Text Processing
//!
//! Formatting helpers, text wrapping, string truncation, duration display, and Levenshtein edit distance.

use std::time::Duration;

/// Computes Levenshtein edit distance between two strings.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

/// Suggests the closest matching candidate for an unknown input name.
pub fn suggest_candidate<'a>(input: &str, candidates: &[&'a str]) -> Option<&'a str> {
    candidates
        .iter()
        .map(|&c| (c, levenshtein_distance(input, c)))
        .filter(|&(_, dist)| dist <= 3)
        .min_by_key(|&(_, dist)| dist)
        .map(|(c, _)| c)
}

/// Truncates string to `max_len` appending `"..."` if truncated.
pub fn truncate_ellipsis(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        "...".to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Formats elapsed duration into human-readable representation.
pub fn format_elapsed(d: Duration) -> String {
    let total_secs = d.as_secs_f64();
    if total_secs < 0.001 {
        format!("{:.2} µs", d.as_nanos() as f64 / 1_000.0)
    } else if total_secs < 1.0 {
        format!("{:.2} ms", d.as_nanos() as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", total_secs)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
