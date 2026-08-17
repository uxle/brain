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

    #[test]
    fn test_transform_suite_2() {
        let s = "hello world_2";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_2"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_2"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_2"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-2"));

        let trans = transliterate_ascii("München Straße «test_2»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_2 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_2", &banned, "***");
        assert_eq!(censored, "this is *** words_2");

        let mut rng = TextRng::new(2 as u64);
        let perturbed = char_perturbation("abcdef_2", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_3() {
        let s = "hello world_3";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_3"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_3"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_3"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-3"));

        let trans = transliterate_ascii("München Straße «test_3»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_3 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_3", &banned, "***");
        assert_eq!(censored, "this is *** words_3");

        let mut rng = TextRng::new(3 as u64);
        let perturbed = char_perturbation("abcdef_3", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_4() {
        let s = "hello world_4";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_4"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_4"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_4"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-4"));

        let trans = transliterate_ascii("München Straße «test_4»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_4 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_4", &banned, "***");
        assert_eq!(censored, "this is *** words_4");

        let mut rng = TextRng::new(4 as u64);
        let perturbed = char_perturbation("abcdef_4", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_5() {
        let s = "hello world_5";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_5"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_5"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_5"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-5"));

        let trans = transliterate_ascii("München Straße «test_5»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_5 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_5", &banned, "***");
        assert_eq!(censored, "this is *** words_5");

        let mut rng = TextRng::new(5 as u64);
        let perturbed = char_perturbation("abcdef_5", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_6() {
        let s = "hello world_6";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_6"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_6"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_6"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-6"));

        let trans = transliterate_ascii("München Straße «test_6»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_6 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_6", &banned, "***");
        assert_eq!(censored, "this is *** words_6");

        let mut rng = TextRng::new(6 as u64);
        let perturbed = char_perturbation("abcdef_6", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_7() {
        let s = "hello world_7";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_7"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_7"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_7"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-7"));

        let trans = transliterate_ascii("München Straße «test_7»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_7 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_7", &banned, "***");
        assert_eq!(censored, "this is *** words_7");

        let mut rng = TextRng::new(7 as u64);
        let perturbed = char_perturbation("abcdef_7", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_8() {
        let s = "hello world_8";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_8"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_8"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_8"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-8"));

        let trans = transliterate_ascii("München Straße «test_8»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_8 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_8", &banned, "***");
        assert_eq!(censored, "this is *** words_8");

        let mut rng = TextRng::new(8 as u64);
        let perturbed = char_perturbation("abcdef_8", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_9() {
        let s = "hello world_9";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_9"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_9"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_9"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-9"));

        let trans = transliterate_ascii("München Straße «test_9»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_9 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_9", &banned, "***");
        assert_eq!(censored, "this is *** words_9");

        let mut rng = TextRng::new(9 as u64);
        let perturbed = char_perturbation("abcdef_9", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_10() {
        let s = "hello world_10";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_10"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_10"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_10"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-10"));

        let trans = transliterate_ascii("München Straße «test_10»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_10 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_10", &banned, "***");
        assert_eq!(censored, "this is *** words_10");

        let mut rng = TextRng::new(10 as u64);
        let perturbed = char_perturbation("abcdef_10", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_11() {
        let s = "hello world_11";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_11"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_11"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_11"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-11"));

        let trans = transliterate_ascii("München Straße «test_11»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_11 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_11", &banned, "***");
        assert_eq!(censored, "this is *** words_11");

        let mut rng = TextRng::new(11 as u64);
        let perturbed = char_perturbation("abcdef_11", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_12() {
        let s = "hello world_12";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_12"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_12"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_12"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-12"));

        let trans = transliterate_ascii("München Straße «test_12»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_12 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_12", &banned, "***");
        assert_eq!(censored, "this is *** words_12");

        let mut rng = TextRng::new(12 as u64);
        let perturbed = char_perturbation("abcdef_12", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_13() {
        let s = "hello world_13";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_13"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_13"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_13"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-13"));

        let trans = transliterate_ascii("München Straße «test_13»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_13 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_13", &banned, "***");
        assert_eq!(censored, "this is *** words_13");

        let mut rng = TextRng::new(13 as u64);
        let perturbed = char_perturbation("abcdef_13", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_14() {
        let s = "hello world_14";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_14"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_14"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_14"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-14"));

        let trans = transliterate_ascii("München Straße «test_14»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_14 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_14", &banned, "***");
        assert_eq!(censored, "this is *** words_14");

        let mut rng = TextRng::new(14 as u64);
        let perturbed = char_perturbation("abcdef_14", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_15() {
        let s = "hello world_15";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_15"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_15"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_15"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-15"));

        let trans = transliterate_ascii("München Straße «test_15»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_15 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_15", &banned, "***");
        assert_eq!(censored, "this is *** words_15");

        let mut rng = TextRng::new(15 as u64);
        let perturbed = char_perturbation("abcdef_15", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_16() {
        let s = "hello world_16";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_16"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_16"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_16"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-16"));

        let trans = transliterate_ascii("München Straße «test_16»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_16 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_16", &banned, "***");
        assert_eq!(censored, "this is *** words_16");

        let mut rng = TextRng::new(16 as u64);
        let perturbed = char_perturbation("abcdef_16", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_17() {
        let s = "hello world_17";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_17"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_17"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_17"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-17"));

        let trans = transliterate_ascii("München Straße «test_17»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_17 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_17", &banned, "***");
        assert_eq!(censored, "this is *** words_17");

        let mut rng = TextRng::new(17 as u64);
        let perturbed = char_perturbation("abcdef_17", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_18() {
        let s = "hello world_18";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_18"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_18"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_18"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-18"));

        let trans = transliterate_ascii("München Straße «test_18»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_18 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_18", &banned, "***");
        assert_eq!(censored, "this is *** words_18");

        let mut rng = TextRng::new(18 as u64);
        let perturbed = char_perturbation("abcdef_18", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_19() {
        let s = "hello world_19";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_19"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_19"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_19"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-19"));

        let trans = transliterate_ascii("München Straße «test_19»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_19 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_19", &banned, "***");
        assert_eq!(censored, "this is *** words_19");

        let mut rng = TextRng::new(19 as u64);
        let perturbed = char_perturbation("abcdef_19", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_20() {
        let s = "hello world_20";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_20"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_20"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_20"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-20"));

        let trans = transliterate_ascii("München Straße «test_20»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_20 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_20", &banned, "***");
        assert_eq!(censored, "this is *** words_20");

        let mut rng = TextRng::new(20 as u64);
        let perturbed = char_perturbation("abcdef_20", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_21() {
        let s = "hello world_21";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_21"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_21"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_21"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-21"));

        let trans = transliterate_ascii("München Straße «test_21»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_21 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_21", &banned, "***");
        assert_eq!(censored, "this is *** words_21");

        let mut rng = TextRng::new(21 as u64);
        let perturbed = char_perturbation("abcdef_21", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_22() {
        let s = "hello world_22";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_22"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_22"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_22"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-22"));

        let trans = transliterate_ascii("München Straße «test_22»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_22 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_22", &banned, "***");
        assert_eq!(censored, "this is *** words_22");

        let mut rng = TextRng::new(22 as u64);
        let perturbed = char_perturbation("abcdef_22", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_23() {
        let s = "hello world_23";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_23"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_23"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_23"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-23"));

        let trans = transliterate_ascii("München Straße «test_23»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_23 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_23", &banned, "***");
        assert_eq!(censored, "this is *** words_23");

        let mut rng = TextRng::new(23 as u64);
        let perturbed = char_perturbation("abcdef_23", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_24() {
        let s = "hello world_24";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_24"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_24"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_24"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-24"));

        let trans = transliterate_ascii("München Straße «test_24»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_24 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_24", &banned, "***");
        assert_eq!(censored, "this is *** words_24");

        let mut rng = TextRng::new(24 as u64);
        let perturbed = char_perturbation("abcdef_24", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_25() {
        let s = "hello world_25";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_25"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_25"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_25"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-25"));

        let trans = transliterate_ascii("München Straße «test_25»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_25 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_25", &banned, "***");
        assert_eq!(censored, "this is *** words_25");

        let mut rng = TextRng::new(25 as u64);
        let perturbed = char_perturbation("abcdef_25", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_26() {
        let s = "hello world_26";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_26"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_26"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_26"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-26"));

        let trans = transliterate_ascii("München Straße «test_26»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_26 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_26", &banned, "***");
        assert_eq!(censored, "this is *** words_26");

        let mut rng = TextRng::new(26 as u64);
        let perturbed = char_perturbation("abcdef_26", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_27() {
        let s = "hello world_27";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_27"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_27"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_27"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-27"));

        let trans = transliterate_ascii("München Straße «test_27»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_27 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_27", &banned, "***");
        assert_eq!(censored, "this is *** words_27");

        let mut rng = TextRng::new(27 as u64);
        let perturbed = char_perturbation("abcdef_27", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_28() {
        let s = "hello world_28";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_28"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_28"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_28"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-28"));

        let trans = transliterate_ascii("München Straße «test_28»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_28 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_28", &banned, "***");
        assert_eq!(censored, "this is *** words_28");

        let mut rng = TextRng::new(28 as u64);
        let perturbed = char_perturbation("abcdef_28", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_29() {
        let s = "hello world_29";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_29"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_29"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_29"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-29"));

        let trans = transliterate_ascii("München Straße «test_29»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_29 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_29", &banned, "***");
        assert_eq!(censored, "this is *** words_29");

        let mut rng = TextRng::new(29 as u64);
        let perturbed = char_perturbation("abcdef_29", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_30() {
        let s = "hello world_30";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_30"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_30"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_30"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-30"));

        let trans = transliterate_ascii("München Straße «test_30»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_30 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_30", &banned, "***");
        assert_eq!(censored, "this is *** words_30");

        let mut rng = TextRng::new(30 as u64);
        let perturbed = char_perturbation("abcdef_30", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_31() {
        let s = "hello world_31";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_31"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_31"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_31"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-31"));

        let trans = transliterate_ascii("München Straße «test_31»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_31 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_31", &banned, "***");
        assert_eq!(censored, "this is *** words_31");

        let mut rng = TextRng::new(31 as u64);
        let perturbed = char_perturbation("abcdef_31", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_32() {
        let s = "hello world_32";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_32"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_32"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_32"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-32"));

        let trans = transliterate_ascii("München Straße «test_32»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_32 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_32", &banned, "***");
        assert_eq!(censored, "this is *** words_32");

        let mut rng = TextRng::new(32 as u64);
        let perturbed = char_perturbation("abcdef_32", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_33() {
        let s = "hello world_33";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_33"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_33"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_33"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-33"));

        let trans = transliterate_ascii("München Straße «test_33»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_33 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_33", &banned, "***");
        assert_eq!(censored, "this is *** words_33");

        let mut rng = TextRng::new(33 as u64);
        let perturbed = char_perturbation("abcdef_33", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_34() {
        let s = "hello world_34";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_34"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_34"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_34"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-34"));

        let trans = transliterate_ascii("München Straße «test_34»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_34 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_34", &banned, "***");
        assert_eq!(censored, "this is *** words_34");

        let mut rng = TextRng::new(34 as u64);
        let perturbed = char_perturbation("abcdef_34", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_35() {
        let s = "hello world_35";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_35"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_35"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_35"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-35"));

        let trans = transliterate_ascii("München Straße «test_35»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_35 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_35", &banned, "***");
        assert_eq!(censored, "this is *** words_35");

        let mut rng = TextRng::new(35 as u64);
        let perturbed = char_perturbation("abcdef_35", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_36() {
        let s = "hello world_36";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_36"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_36"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_36"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-36"));

        let trans = transliterate_ascii("München Straße «test_36»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_36 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_36", &banned, "***");
        assert_eq!(censored, "this is *** words_36");

        let mut rng = TextRng::new(36 as u64);
        let perturbed = char_perturbation("abcdef_36", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_37() {
        let s = "hello world_37";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_37"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_37"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_37"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-37"));

        let trans = transliterate_ascii("München Straße «test_37»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_37 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_37", &banned, "***");
        assert_eq!(censored, "this is *** words_37");

        let mut rng = TextRng::new(37 as u64);
        let perturbed = char_perturbation("abcdef_37", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_38() {
        let s = "hello world_38";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_38"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_38"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_38"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-38"));

        let trans = transliterate_ascii("München Straße «test_38»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_38 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_38", &banned, "***");
        assert_eq!(censored, "this is *** words_38");

        let mut rng = TextRng::new(38 as u64);
        let perturbed = char_perturbation("abcdef_38", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_39() {
        let s = "hello world_39";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_39"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_39"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_39"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-39"));

        let trans = transliterate_ascii("München Straße «test_39»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_39 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_39", &banned, "***");
        assert_eq!(censored, "this is *** words_39");

        let mut rng = TextRng::new(39 as u64);
        let perturbed = char_perturbation("abcdef_39", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_40() {
        let s = "hello world_40";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_40"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_40"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_40"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-40"));

        let trans = transliterate_ascii("München Straße «test_40»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_40 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_40", &banned, "***");
        assert_eq!(censored, "this is *** words_40");

        let mut rng = TextRng::new(40 as u64);
        let perturbed = char_perturbation("abcdef_40", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_41() {
        let s = "hello world_41";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_41"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_41"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_41"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-41"));

        let trans = transliterate_ascii("München Straße «test_41»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_41 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_41", &banned, "***");
        assert_eq!(censored, "this is *** words_41");

        let mut rng = TextRng::new(41 as u64);
        let perturbed = char_perturbation("abcdef_41", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_42() {
        let s = "hello world_42";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_42"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_42"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_42"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-42"));

        let trans = transliterate_ascii("München Straße «test_42»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_42 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_42", &banned, "***");
        assert_eq!(censored, "this is *** words_42");

        let mut rng = TextRng::new(42 as u64);
        let perturbed = char_perturbation("abcdef_42", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_43() {
        let s = "hello world_43";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_43"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_43"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_43"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-43"));

        let trans = transliterate_ascii("München Straße «test_43»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_43 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_43", &banned, "***");
        assert_eq!(censored, "this is *** words_43");

        let mut rng = TextRng::new(43 as u64);
        let perturbed = char_perturbation("abcdef_43", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_44() {
        let s = "hello world_44";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_44"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_44"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_44"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-44"));

        let trans = transliterate_ascii("München Straße «test_44»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_44 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_44", &banned, "***");
        assert_eq!(censored, "this is *** words_44");

        let mut rng = TextRng::new(44 as u64);
        let perturbed = char_perturbation("abcdef_44", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_45() {
        let s = "hello world_45";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_45"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_45"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_45"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-45"));

        let trans = transliterate_ascii("München Straße «test_45»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_45 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_45", &banned, "***");
        assert_eq!(censored, "this is *** words_45");

        let mut rng = TextRng::new(45 as u64);
        let perturbed = char_perturbation("abcdef_45", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_46() {
        let s = "hello world_46";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_46"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_46"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_46"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-46"));

        let trans = transliterate_ascii("München Straße «test_46»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_46 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_46", &banned, "***");
        assert_eq!(censored, "this is *** words_46");

        let mut rng = TextRng::new(46 as u64);
        let perturbed = char_perturbation("abcdef_46", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_47() {
        let s = "hello world_47";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_47"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_47"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_47"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-47"));

        let trans = transliterate_ascii("München Straße «test_47»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_47 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_47", &banned, "***");
        assert_eq!(censored, "this is *** words_47");

        let mut rng = TextRng::new(47 as u64);
        let perturbed = char_perturbation("abcdef_47", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_48() {
        let s = "hello world_48";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_48"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_48"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_48"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-48"));

        let trans = transliterate_ascii("München Straße «test_48»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_48 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_48", &banned, "***");
        assert_eq!(censored, "this is *** words_48");

        let mut rng = TextRng::new(48 as u64);
        let perturbed = char_perturbation("abcdef_48", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_49() {
        let s = "hello world_49";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_49"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_49"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_49"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-49"));

        let trans = transliterate_ascii("München Straße «test_49»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_49 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_49", &banned, "***");
        assert_eq!(censored, "this is *** words_49");

        let mut rng = TextRng::new(49 as u64);
        let perturbed = char_perturbation("abcdef_49", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_50() {
        let s = "hello world_50";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_50"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_50"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_50"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-50"));

        let trans = transliterate_ascii("München Straße «test_50»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_50 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_50", &banned, "***");
        assert_eq!(censored, "this is *** words_50");

        let mut rng = TextRng::new(50 as u64);
        let perturbed = char_perturbation("abcdef_50", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_51() {
        let s = "hello world_51";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_51"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_51"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_51"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-51"));

        let trans = transliterate_ascii("München Straße «test_51»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_51 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_51", &banned, "***");
        assert_eq!(censored, "this is *** words_51");

        let mut rng = TextRng::new(51 as u64);
        let perturbed = char_perturbation("abcdef_51", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_52() {
        let s = "hello world_52";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_52"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_52"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_52"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-52"));

        let trans = transliterate_ascii("München Straße «test_52»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_52 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_52", &banned, "***");
        assert_eq!(censored, "this is *** words_52");

        let mut rng = TextRng::new(52 as u64);
        let perturbed = char_perturbation("abcdef_52", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_53() {
        let s = "hello world_53";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_53"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_53"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_53"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-53"));

        let trans = transliterate_ascii("München Straße «test_53»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_53 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_53", &banned, "***");
        assert_eq!(censored, "this is *** words_53");

        let mut rng = TextRng::new(53 as u64);
        let perturbed = char_perturbation("abcdef_53", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_54() {
        let s = "hello world_54";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_54"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_54"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_54"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-54"));

        let trans = transliterate_ascii("München Straße «test_54»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_54 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_54", &banned, "***");
        assert_eq!(censored, "this is *** words_54");

        let mut rng = TextRng::new(54 as u64);
        let perturbed = char_perturbation("abcdef_54", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_55() {
        let s = "hello world_55";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_55"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_55"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_55"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-55"));

        let trans = transliterate_ascii("München Straße «test_55»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_55 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_55", &banned, "***");
        assert_eq!(censored, "this is *** words_55");

        let mut rng = TextRng::new(55 as u64);
        let perturbed = char_perturbation("abcdef_55", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_56() {
        let s = "hello world_56";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_56"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_56"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_56"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-56"));

        let trans = transliterate_ascii("München Straße «test_56»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_56 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_56", &banned, "***");
        assert_eq!(censored, "this is *** words_56");

        let mut rng = TextRng::new(56 as u64);
        let perturbed = char_perturbation("abcdef_56", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_57() {
        let s = "hello world_57";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_57"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_57"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_57"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-57"));

        let trans = transliterate_ascii("München Straße «test_57»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_57 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_57", &banned, "***");
        assert_eq!(censored, "this is *** words_57");

        let mut rng = TextRng::new(57 as u64);
        let perturbed = char_perturbation("abcdef_57", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_58() {
        let s = "hello world_58";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_58"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_58"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_58"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-58"));

        let trans = transliterate_ascii("München Straße «test_58»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_58 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_58", &banned, "***");
        assert_eq!(censored, "this is *** words_58");

        let mut rng = TextRng::new(58 as u64);
        let perturbed = char_perturbation("abcdef_58", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_59() {
        let s = "hello world_59";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_59"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_59"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_59"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-59"));

        let trans = transliterate_ascii("München Straße «test_59»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_59 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_59", &banned, "***");
        assert_eq!(censored, "this is *** words_59");

        let mut rng = TextRng::new(59 as u64);
        let perturbed = char_perturbation("abcdef_59", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_60() {
        let s = "hello world_60";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_60"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_60"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_60"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-60"));

        let trans = transliterate_ascii("München Straße «test_60»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_60 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_60", &banned, "***");
        assert_eq!(censored, "this is *** words_60");

        let mut rng = TextRng::new(60 as u64);
        let perturbed = char_perturbation("abcdef_60", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_61() {
        let s = "hello world_61";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_61"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_61"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_61"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-61"));

        let trans = transliterate_ascii("München Straße «test_61»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_61 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_61", &banned, "***");
        assert_eq!(censored, "this is *** words_61");

        let mut rng = TextRng::new(61 as u64);
        let perturbed = char_perturbation("abcdef_61", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_62() {
        let s = "hello world_62";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_62"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_62"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_62"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-62"));

        let trans = transliterate_ascii("München Straße «test_62»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_62 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_62", &banned, "***");
        assert_eq!(censored, "this is *** words_62");

        let mut rng = TextRng::new(62 as u64);
        let perturbed = char_perturbation("abcdef_62", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_63() {
        let s = "hello world_63";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_63"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_63"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_63"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-63"));

        let trans = transliterate_ascii("München Straße «test_63»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_63 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_63", &banned, "***");
        assert_eq!(censored, "this is *** words_63");

        let mut rng = TextRng::new(63 as u64);
        let perturbed = char_perturbation("abcdef_63", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_64() {
        let s = "hello world_64";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_64"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_64"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_64"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-64"));

        let trans = transliterate_ascii("München Straße «test_64»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_64 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_64", &banned, "***");
        assert_eq!(censored, "this is *** words_64");

        let mut rng = TextRng::new(64 as u64);
        let perturbed = char_perturbation("abcdef_64", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_65() {
        let s = "hello world_65";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_65"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_65"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_65"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-65"));

        let trans = transliterate_ascii("München Straße «test_65»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_65 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_65", &banned, "***");
        assert_eq!(censored, "this is *** words_65");

        let mut rng = TextRng::new(65 as u64);
        let perturbed = char_perturbation("abcdef_65", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_66() {
        let s = "hello world_66";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_66"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_66"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_66"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-66"));

        let trans = transliterate_ascii("München Straße «test_66»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_66 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_66", &banned, "***");
        assert_eq!(censored, "this is *** words_66");

        let mut rng = TextRng::new(66 as u64);
        let perturbed = char_perturbation("abcdef_66", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_67() {
        let s = "hello world_67";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_67"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_67"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_67"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-67"));

        let trans = transliterate_ascii("München Straße «test_67»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_67 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_67", &banned, "***");
        assert_eq!(censored, "this is *** words_67");

        let mut rng = TextRng::new(67 as u64);
        let perturbed = char_perturbation("abcdef_67", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_68() {
        let s = "hello world_68";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_68"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_68"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_68"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-68"));

        let trans = transliterate_ascii("München Straße «test_68»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_68 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_68", &banned, "***");
        assert_eq!(censored, "this is *** words_68");

        let mut rng = TextRng::new(68 as u64);
        let perturbed = char_perturbation("abcdef_68", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_69() {
        let s = "hello world_69";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_69"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_69"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_69"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-69"));

        let trans = transliterate_ascii("München Straße «test_69»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_69 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_69", &banned, "***");
        assert_eq!(censored, "this is *** words_69");

        let mut rng = TextRng::new(69 as u64);
        let perturbed = char_perturbation("abcdef_69", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_70() {
        let s = "hello world_70";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_70"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_70"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_70"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-70"));

        let trans = transliterate_ascii("München Straße «test_70»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_70 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_70", &banned, "***");
        assert_eq!(censored, "this is *** words_70");

        let mut rng = TextRng::new(70 as u64);
        let perturbed = char_perturbation("abcdef_70", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_71() {
        let s = "hello world_71";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_71"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_71"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_71"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-71"));

        let trans = transliterate_ascii("München Straße «test_71»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_71 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_71", &banned, "***");
        assert_eq!(censored, "this is *** words_71");

        let mut rng = TextRng::new(71 as u64);
        let perturbed = char_perturbation("abcdef_71", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_72() {
        let s = "hello world_72";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_72"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_72"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_72"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-72"));

        let trans = transliterate_ascii("München Straße «test_72»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_72 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_72", &banned, "***");
        assert_eq!(censored, "this is *** words_72");

        let mut rng = TextRng::new(72 as u64);
        let perturbed = char_perturbation("abcdef_72", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_73() {
        let s = "hello world_73";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_73"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_73"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_73"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-73"));

        let trans = transliterate_ascii("München Straße «test_73»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_73 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_73", &banned, "***");
        assert_eq!(censored, "this is *** words_73");

        let mut rng = TextRng::new(73 as u64);
        let perturbed = char_perturbation("abcdef_73", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_74() {
        let s = "hello world_74";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_74"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_74"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_74"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-74"));

        let trans = transliterate_ascii("München Straße «test_74»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_74 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_74", &banned, "***");
        assert_eq!(censored, "this is *** words_74");

        let mut rng = TextRng::new(74 as u64);
        let perturbed = char_perturbation("abcdef_74", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_75() {
        let s = "hello world_75";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_75"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_75"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_75"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-75"));

        let trans = transliterate_ascii("München Straße «test_75»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_75 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_75", &banned, "***");
        assert_eq!(censored, "this is *** words_75");

        let mut rng = TextRng::new(75 as u64);
        let perturbed = char_perturbation("abcdef_75", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_76() {
        let s = "hello world_76";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_76"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_76"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_76"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-76"));

        let trans = transliterate_ascii("München Straße «test_76»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_76 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_76", &banned, "***");
        assert_eq!(censored, "this is *** words_76");

        let mut rng = TextRng::new(76 as u64);
        let perturbed = char_perturbation("abcdef_76", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_77() {
        let s = "hello world_77";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_77"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_77"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_77"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-77"));

        let trans = transliterate_ascii("München Straße «test_77»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_77 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_77", &banned, "***");
        assert_eq!(censored, "this is *** words_77");

        let mut rng = TextRng::new(77 as u64);
        let perturbed = char_perturbation("abcdef_77", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_78() {
        let s = "hello world_78";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_78"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_78"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_78"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-78"));

        let trans = transliterate_ascii("München Straße «test_78»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_78 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_78", &banned, "***");
        assert_eq!(censored, "this is *** words_78");

        let mut rng = TextRng::new(78 as u64);
        let perturbed = char_perturbation("abcdef_78", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_79() {
        let s = "hello world_79";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_79"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_79"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_79"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-79"));

        let trans = transliterate_ascii("München Straße «test_79»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_79 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_79", &banned, "***");
        assert_eq!(censored, "this is *** words_79");

        let mut rng = TextRng::new(79 as u64);
        let perturbed = char_perturbation("abcdef_79", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_80() {
        let s = "hello world_80";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_80"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_80"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_80"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-80"));

        let trans = transliterate_ascii("München Straße «test_80»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_80 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_80", &banned, "***");
        assert_eq!(censored, "this is *** words_80");

        let mut rng = TextRng::new(80 as u64);
        let perturbed = char_perturbation("abcdef_80", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_81() {
        let s = "hello world_81";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_81"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_81"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_81"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-81"));

        let trans = transliterate_ascii("München Straße «test_81»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_81 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_81", &banned, "***");
        assert_eq!(censored, "this is *** words_81");

        let mut rng = TextRng::new(81 as u64);
        let perturbed = char_perturbation("abcdef_81", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_82() {
        let s = "hello world_82";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_82"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_82"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_82"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-82"));

        let trans = transliterate_ascii("München Straße «test_82»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_82 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_82", &banned, "***");
        assert_eq!(censored, "this is *** words_82");

        let mut rng = TextRng::new(82 as u64);
        let perturbed = char_perturbation("abcdef_82", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_83() {
        let s = "hello world_83";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_83"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_83"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_83"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-83"));

        let trans = transliterate_ascii("München Straße «test_83»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_83 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_83", &banned, "***");
        assert_eq!(censored, "this is *** words_83");

        let mut rng = TextRng::new(83 as u64);
        let perturbed = char_perturbation("abcdef_83", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_84() {
        let s = "hello world_84";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_84"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_84"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_84"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-84"));

        let trans = transliterate_ascii("München Straße «test_84»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_84 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_84", &banned, "***");
        assert_eq!(censored, "this is *** words_84");

        let mut rng = TextRng::new(84 as u64);
        let perturbed = char_perturbation("abcdef_84", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_85() {
        let s = "hello world_85";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_85"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_85"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_85"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-85"));

        let trans = transliterate_ascii("München Straße «test_85»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_85 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_85", &banned, "***");
        assert_eq!(censored, "this is *** words_85");

        let mut rng = TextRng::new(85 as u64);
        let perturbed = char_perturbation("abcdef_85", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_86() {
        let s = "hello world_86";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_86"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_86"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_86"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-86"));

        let trans = transliterate_ascii("München Straße «test_86»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_86 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_86", &banned, "***");
        assert_eq!(censored, "this is *** words_86");

        let mut rng = TextRng::new(86 as u64);
        let perturbed = char_perturbation("abcdef_86", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_87() {
        let s = "hello world_87";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_87"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_87"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_87"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-87"));

        let trans = transliterate_ascii("München Straße «test_87»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_87 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_87", &banned, "***");
        assert_eq!(censored, "this is *** words_87");

        let mut rng = TextRng::new(87 as u64);
        let perturbed = char_perturbation("abcdef_87", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_88() {
        let s = "hello world_88";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_88"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_88"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_88"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-88"));

        let trans = transliterate_ascii("München Straße «test_88»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_88 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_88", &banned, "***");
        assert_eq!(censored, "this is *** words_88");

        let mut rng = TextRng::new(88 as u64);
        let perturbed = char_perturbation("abcdef_88", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_89() {
        let s = "hello world_89";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_89"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_89"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_89"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-89"));

        let trans = transliterate_ascii("München Straße «test_89»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_89 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_89", &banned, "***");
        assert_eq!(censored, "this is *** words_89");

        let mut rng = TextRng::new(89 as u64);
        let perturbed = char_perturbation("abcdef_89", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_90() {
        let s = "hello world_90";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_90"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_90"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_90"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-90"));

        let trans = transliterate_ascii("München Straße «test_90»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_90 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_90", &banned, "***");
        assert_eq!(censored, "this is *** words_90");

        let mut rng = TextRng::new(90 as u64);
        let perturbed = char_perturbation("abcdef_90", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_91() {
        let s = "hello world_91";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_91"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_91"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_91"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-91"));

        let trans = transliterate_ascii("München Straße «test_91»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_91 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_91", &banned, "***");
        assert_eq!(censored, "this is *** words_91");

        let mut rng = TextRng::new(91 as u64);
        let perturbed = char_perturbation("abcdef_91", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_92() {
        let s = "hello world_92";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_92"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_92"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_92"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-92"));

        let trans = transliterate_ascii("München Straße «test_92»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_92 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_92", &banned, "***");
        assert_eq!(censored, "this is *** words_92");

        let mut rng = TextRng::new(92 as u64);
        let perturbed = char_perturbation("abcdef_92", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_93() {
        let s = "hello world_93";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_93"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_93"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_93"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-93"));

        let trans = transliterate_ascii("München Straße «test_93»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_93 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_93", &banned, "***");
        assert_eq!(censored, "this is *** words_93");

        let mut rng = TextRng::new(93 as u64);
        let perturbed = char_perturbation("abcdef_93", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_94() {
        let s = "hello world_94";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_94"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_94"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_94"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-94"));

        let trans = transliterate_ascii("München Straße «test_94»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_94 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_94", &banned, "***");
        assert_eq!(censored, "this is *** words_94");

        let mut rng = TextRng::new(94 as u64);
        let perturbed = char_perturbation("abcdef_94", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_95() {
        let s = "hello world_95";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_95"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_95"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_95"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-95"));

        let trans = transliterate_ascii("München Straße «test_95»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_95 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_95", &banned, "***");
        assert_eq!(censored, "this is *** words_95");

        let mut rng = TextRng::new(95 as u64);
        let perturbed = char_perturbation("abcdef_95", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_96() {
        let s = "hello world_96";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_96"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_96"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_96"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-96"));

        let trans = transliterate_ascii("München Straße «test_96»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_96 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_96", &banned, "***");
        assert_eq!(censored, "this is *** words_96");

        let mut rng = TextRng::new(96 as u64);
        let perturbed = char_perturbation("abcdef_96", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_97() {
        let s = "hello world_97";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_97"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_97"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_97"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-97"));

        let trans = transliterate_ascii("München Straße «test_97»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_97 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_97", &banned, "***");
        assert_eq!(censored, "this is *** words_97");

        let mut rng = TextRng::new(97 as u64);
        let perturbed = char_perturbation("abcdef_97", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_98() {
        let s = "hello world_98";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_98"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_98"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_98"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-98"));

        let trans = transliterate_ascii("München Straße «test_98»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_98 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_98", &banned, "***");
        assert_eq!(censored, "this is *** words_98");

        let mut rng = TextRng::new(98 as u64);
        let perturbed = char_perturbation("abcdef_98", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_99() {
        let s = "hello world_99";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_99"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_99"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_99"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-99"));

        let trans = transliterate_ascii("München Straße «test_99»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_99 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_99", &banned, "***");
        assert_eq!(censored, "this is *** words_99");

        let mut rng = TextRng::new(99 as u64);
        let perturbed = char_perturbation("abcdef_99", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_100() {
        let s = "hello world_100";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_100"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_100"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_100"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-100"));

        let trans = transliterate_ascii("München Straße «test_100»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_100 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_100", &banned, "***");
        assert_eq!(censored, "this is *** words_100");

        let mut rng = TextRng::new(100 as u64);
        let perturbed = char_perturbation("abcdef_100", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_101() {
        let s = "hello world_101";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_101"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_101"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_101"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-101"));

        let trans = transliterate_ascii("München Straße «test_101»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_101 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_101", &banned, "***");
        assert_eq!(censored, "this is *** words_101");

        let mut rng = TextRng::new(101 as u64);
        let perturbed = char_perturbation("abcdef_101", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_102() {
        let s = "hello world_102";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_102"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_102"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_102"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-102"));

        let trans = transliterate_ascii("München Straße «test_102»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_102 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_102", &banned, "***");
        assert_eq!(censored, "this is *** words_102");

        let mut rng = TextRng::new(102 as u64);
        let perturbed = char_perturbation("abcdef_102", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_103() {
        let s = "hello world_103";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_103"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_103"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_103"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-103"));

        let trans = transliterate_ascii("München Straße «test_103»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_103 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_103", &banned, "***");
        assert_eq!(censored, "this is *** words_103");

        let mut rng = TextRng::new(103 as u64);
        let perturbed = char_perturbation("abcdef_103", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_104() {
        let s = "hello world_104";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_104"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_104"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_104"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-104"));

        let trans = transliterate_ascii("München Straße «test_104»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_104 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_104", &banned, "***");
        assert_eq!(censored, "this is *** words_104");

        let mut rng = TextRng::new(104 as u64);
        let perturbed = char_perturbation("abcdef_104", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_105() {
        let s = "hello world_105";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_105"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_105"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_105"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-105"));

        let trans = transliterate_ascii("München Straße «test_105»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_105 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_105", &banned, "***");
        assert_eq!(censored, "this is *** words_105");

        let mut rng = TextRng::new(105 as u64);
        let perturbed = char_perturbation("abcdef_105", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_106() {
        let s = "hello world_106";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_106"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_106"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_106"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-106"));

        let trans = transliterate_ascii("München Straße «test_106»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_106 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_106", &banned, "***");
        assert_eq!(censored, "this is *** words_106");

        let mut rng = TextRng::new(106 as u64);
        let perturbed = char_perturbation("abcdef_106", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_107() {
        let s = "hello world_107";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_107"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_107"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_107"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-107"));

        let trans = transliterate_ascii("München Straße «test_107»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_107 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_107", &banned, "***");
        assert_eq!(censored, "this is *** words_107");

        let mut rng = TextRng::new(107 as u64);
        let perturbed = char_perturbation("abcdef_107", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_108() {
        let s = "hello world_108";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_108"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_108"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_108"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-108"));

        let trans = transliterate_ascii("München Straße «test_108»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_108 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_108", &banned, "***");
        assert_eq!(censored, "this is *** words_108");

        let mut rng = TextRng::new(108 as u64);
        let perturbed = char_perturbation("abcdef_108", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_109() {
        let s = "hello world_109";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_109"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_109"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_109"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-109"));

        let trans = transliterate_ascii("München Straße «test_109»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_109 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_109", &banned, "***");
        assert_eq!(censored, "this is *** words_109");

        let mut rng = TextRng::new(109 as u64);
        let perturbed = char_perturbation("abcdef_109", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_110() {
        let s = "hello world_110";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_110"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_110"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_110"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-110"));

        let trans = transliterate_ascii("München Straße «test_110»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_110 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_110", &banned, "***");
        assert_eq!(censored, "this is *** words_110");

        let mut rng = TextRng::new(110 as u64);
        let perturbed = char_perturbation("abcdef_110", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_111() {
        let s = "hello world_111";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_111"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_111"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_111"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-111"));

        let trans = transliterate_ascii("München Straße «test_111»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_111 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_111", &banned, "***");
        assert_eq!(censored, "this is *** words_111");

        let mut rng = TextRng::new(111 as u64);
        let perturbed = char_perturbation("abcdef_111", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }

    #[test]
    fn test_transform_suite_112() {
        let s = "hello world_112";
        assert_eq!(case_transform(s, CaseKind::Upper), format!("HELLO WORLD_112"));
        assert_eq!(case_transform(s, CaseKind::Title), format!("Hello World_112"));
        assert_eq!(case_transform(s, CaseKind::Snake), format!("hello_world_112"));
        assert_eq!(case_transform(s, CaseKind::Kebab), format!("hello-world-112"));

        let trans = transliterate_ascii("München Straße «test_112»");
        assert!(trans.contains("Munchen"));
        assert!(trans.contains("Strasse"));

        let norm_p = normalize_punctuation("hello_112 - test");
        assert!(norm_p.contains("hello"));

        let mut banned = HashSet::new();
        banned.insert("bad".to_string());
        let censored = censor_words("this is bad words_112", &banned, "***");
        assert_eq!(censored, "this is *** words_112");

        let mut rng = TextRng::new(112 as u64);
        let perturbed = char_perturbation("abcdef_112", 0.2, &mut rng);
        assert!(!perturbed.is_empty());

        let pipe = TextPipeline::new().add_step(|s| s.to_uppercase());
        assert_eq!(pipe.run("abc"), "ABC");
    }
}
