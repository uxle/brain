//! # Text Transformations, Casing, Transliteration, and Pipelines
//!
//! Case conversions, ASCII transliteration, regex-free pattern substitutions, profanity masking, and chainable pipelines.
#![allow(missing_docs, unused_imports, unused_variables, dead_code, unused_mut, unused_comparisons, clippy::all)]

use crate::utils::TextRng;
use std::collections::HashSet;

/// Text case conversion target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaseKind {
    /// lowercase
    Lower,
    /// UPPERCASE
    Upper,
    /// Title Case
    Title,
    /// camelCase
    Camel,
    /// snake_case
    Snake,
    /// kebab-case
    Kebab,
}

/// Applies structured case conversion to a string.
pub fn case_transform(text: &str, kind: CaseKind) -> String {
    match kind {
        CaseKind::Lower => text.to_lowercase(),
        CaseKind::Upper => text.to_uppercase(),
        CaseKind::Title => {
            let mut result = Vec::new();
            for word in text.split_whitespace() {
                let mut chars = word.chars();
                let first = chars.next().map(|c| c.to_uppercase().to_string()).unwrap_or_default();
                let rest: String = chars.as_str().to_lowercase();
                result.push(format!("{}{}", first, rest));
            }
            result.join(" ")
        }
        CaseKind::Camel => {
            let words: Vec<&str> = text.split(|c: char| c.is_whitespace() || c == '_' || c == '-').filter(|w| !w.is_empty()).collect();
            let mut result = String::new();
            for (i, &w) in words.iter().enumerate() {
                if i == 0 {
                    result.push_str(&w.to_lowercase());
                } else {
                    let mut chars = w.chars();
                    let first = chars.next().map(|c| c.to_uppercase().to_string()).unwrap_or_default();
                    let rest: String = chars.as_str().to_lowercase();
                    result.push_str(&first);
                    result.push_str(&rest);
                }
            }
            result
        }
        CaseKind::Snake => {
            let words: Vec<&str> = text.split(|c: char| c.is_whitespace() || c == '_' || c == '-').filter(|w| !w.is_empty()).collect();
            words.iter().map(|w| w.to_lowercase()).collect::<Vec<String>>().join("_")
        }
        CaseKind::Kebab => {
            let words: Vec<&str> = text.split(|c: char| c.is_whitespace() || c == '_' || c == '-').filter(|w| !w.is_empty()).collect();
            words.iter().map(|w| w.to_lowercase()).collect::<Vec<String>>().join("-")
        }
    }
}

/// Transliterates common Unicode characters to approximate ASCII representations.
pub fn transliterate_ascii(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'ß' => result.push_str("ss"),
            'æ' => result.push_str("ae"),
            'Æ' => result.push_str("AE"),
            'œ' => result.push_str("oe"),
            'Œ' => result.push_str("OE"),
            'ø' => result.push('o'),
            'Ø' => result.push('O'),
            '©' => result.push_str("(c)"),
            '®' => result.push_str("(r)"),
            '™' => result.push_str("TM"),
            '«' | '»' | '“' | '”' => result.push('"'),
            '‘' | '’' => result.push('\''),
            '—' | '–' => result.push('-'),
            '…' => result.push_str("..."),
            _ => {
                let stripped = crate::utils::unicode_helpers::strip_accents(&c.to_string());
                result.push_str(&stripped);
            }
        }
    }
    result
}

/// Normalizes punctuation variants (curled quotes, dashes, ellipses) to standard ASCII equivalents.
pub fn normalize_punctuation(text: &str) -> String {
    text.replace('“', "\"")
        .replace('”', "\"")
        .replace('‘', "'")
        .replace('’', "'")
        .replace('—', "-")
        .replace('–', "-")
        .replace('…', "...")
}

/// Replaces string patterns based on a sequence of rules.
pub fn replace_patterns(text: &str, rules: &[(String, String)]) -> String {
    let mut result = text.to_string();
    for (from, to) in rules {
        result = result.replace(from, to);
    }
    result
}

/// Replaces words in banned set with a censor mask string (e.g. `****`).
pub fn censor_words(text: &str, banned: &HashSet<String>, replacement: &str) -> String {
    let mut words: Vec<String> = Vec::new();
    for w in text.split_whitespace() {
        let cleaned: String = w.chars().filter(|c| c.is_alphabetic()).collect();
        if banned.contains(&cleaned.to_lowercase()) {
            words.push(replacement.to_string());
        } else {
            words.push(w.to_string());
        }
    }
    words.join(" ")
}

/// Applies character-level random perturbation (typo injection).
pub fn char_perturbation(text: &str, prob: f32, rng: &mut TextRng) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        if rng.next_f32() < prob && c.is_ascii_alphabetic() {
            let offset = (rng.gen_range(5) as u8) % 26;
            let perturbed = if c.is_ascii_lowercase() {
                (b'a' + (c as u8 - b'a' + offset + 1) % 26) as char
            } else {
                (b'A' + (c as u8 - b'A' + offset + 1) % 26) as char
            };
            result.push(perturbed);
        } else {
            result.push(c);
        }
    }
    result
}

/// Chainable, configurable text transformation pipeline.
#[derive(Debug, Clone, Default)]
pub struct TextPipeline {
    transforms: Vec<Box<fn(&str) -> String>>,
}

impl TextPipeline {
    /// Creates an empty pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a transformation function to the pipeline.
    pub fn add_step(mut self, step: fn(&str) -> String) -> Self {
        self.transforms.push(Box::new(step));
        self
    }

    /// Executes the pipeline sequentially over an input string.
    pub fn run(&self, input: &str) -> String {
        let mut curr = input.to_string();
        for t in &self.transforms {
            curr = t(&curr);
        }
        curr
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
    fn test_transform_suite_1() {
        let s = "hello world_1";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_1"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_1"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_1"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-1"));

        let trans = transliterate_ascii("München Straße «test_1»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_1 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_1", &banned, "***");
        assert_eq!(censored, "this is *** words_1");

        let mut rng = TextRng::new(1 as u64);
        let perturbed = char_perturbation("abcdef_1", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }
}
