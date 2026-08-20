//! # Tokenizer Integration Tests
use brain_text::*;

#[test]
fn test_word_tokenizer_and_vocab() {
    let text = "the quick brown fox jumps over the lazy dog";
    let words = split_ws(text);
    assert_eq!(words.len(), 9);

    let mut vocab = Vocab::new();
    for word in &words {
        vocab.insert(word);
    }
    assert_eq!(vocab.len(), 8); // 'the' is deduplicated

    let dist = levenshtein_distance("kitten", "sitting");
    assert_eq!(dist, 3);
}

#[test]
fn test_bleu_score_exact_match() {
    let reference = vec![
        "the".into(),
        "cat".into(),
        "sat".into(),
        "on".into(),
        "the".into(),
        "mat".into(),
    ];
    let candidate = reference.clone();

    let score = bleu_score(&reference, &candidate, 4, true);
    assert!(
        (score - 1.0).abs() < 1e-4,
        "Exact match BLEU must be 1.0, got {}",
        score
    );
}
