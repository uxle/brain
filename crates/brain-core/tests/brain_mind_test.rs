use std::fs;
use std::path::PathBuf;

use brain_core::brain_mind::BrainMind;

fn temp_file(name: &str, content: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("brain_mind_test_{}_{}", std::process::id(), name));
    fs::write(&path, content).unwrap();
    path
}

fn fresh_brain() -> BrainMind {
    BrainMind::new("test-brain", 16)
}

#[test]
fn colon_prose_lines_are_not_indexed_as_facts() {
    let mut brain = fresh_brain();
    let path = temp_file(
        "colon_prose.txt",
        "TYPE: grammar\nNAME: neither\nDEFINITION: not either\nEXAMPLES:\n  correct: I don't have any money.\n  incorrect: I don't have no money.\nEND\nhardly ever = 5%\n",
    );
    brain.teach_file(&path).unwrap();

    let resp = brain.talk("what is the correct answer and why?");
    assert!(
        !resp.contains("money"),
        "colon-prose line was indexed as a fact: {}",
        resp
    );
    assert!(resp.contains("don't know"), "expected humble response, got: {}", resp);
}

#[test]
fn typed_section_facts_are_indexed() {
    let mut brain = fresh_brain();
    let path = temp_file(
        "typed.txt",
        "TYPE: vocabulary\nNAME: bird\nDEFINITION: a feathered flying animal\nEND\n",
    );
    brain.teach_file(&path).unwrap();

    let resp = brain.talk("what is a bird");
    assert!(
        resp.contains("feathered flying animal"),
        "typed-section definition not retrieved: {}",
        resp
    );
}

#[test]
fn analogy_question_not_hijacked_by_unrelated_fact() {
    let mut brain = fresh_brain();
    let path = temp_file(
        "analogy.txt",
        "TYPE: grammar\nNAME: neither\nDEFINITION: not either\nEND\n\
         TYPE: vocabulary\nNAME: nest\nDEFINITION: a home built by birds\nEND\n",
    );
    brain.teach_file(&path).unwrap();

    let resp = brain.talk("In the analogy birds:nest, what is the correct answer and why?");
    assert!(
        !resp.contains("neither") && !resp.contains("not either"),
        "question hijacked by unrelated fact: {}",
        resp
    );
    assert!(resp.contains("don't know"), "expected humble response, got: {}", resp);
}

#[test]
fn gibberish_question_gets_humble_response() {
    let mut brain = fresh_brain();
    let resp = brain.talk("what is naun");
    assert!(resp.contains("don't know"), "got: {}", resp);
    assert!(resp.contains("naun"), "expected keyword in response: {}", resp);
}

#[test]
fn neural_adam_reduces_loss_and_learns() {
    let mut brain = fresh_brain();
    let phrase = "the cat sat on the mat and the dog ran home";
    let first = brain.neural_adam_train_sequence(phrase, 0.01);
    let mut last = first;
    for _ in 0..200 {
        last = brain.neural_adam_train_sequence(phrase, 0.01);
    }
    assert!(
        last < first,
        "Adam did not reduce loss: {:.4} -> {:.4}",
        first,
        last
    );
    assert!(
        last < 0.05,
        "Adam stalled far above zero loss: {:.4}",
        last
    );

    let gen = brain.neural_generate("the cat sat", 16, 0.05);
    assert!(
        gen.contains("mat"),
        "model did not learn the phrase from Adam: {:?}",
        gen
    );
}

#[test]
fn neural_train_sequence_changes_weights() {
    let mut brain = fresh_brain();
    let before: Vec<f64> = brain.lm_head.data().to_vec();
    brain.neural_train_sequence("the quick brown fox jumps", 0.01);
    let after: Vec<f64> = brain.lm_head.data().to_vec();
    assert!(
        before != after,
        "weights unchanged - update is not gradient descent"
    );
}

#[test]
fn percentage_math_still_works() {
    let mut brain = fresh_brain();
    let resp = brain.talk("what is 20% of 150?");
    assert!(resp.contains("30"), "percentage math broken: {}", resp);
    assert!(!resp.contains("don't know"), "percentage math fell through: {}", resp);
}

#[test]
fn name_learning_still_works() {
    let mut brain = fresh_brain();
    brain.talk("my name is alice");
    let resp = brain.talk("who am i");
    assert!(resp.contains("alice"), "brain did not remember the name: {}", resp);
}

#[test]
fn newborn_echo_is_gated_to_statements() {
    let mut brain = fresh_brain();
    let resp = brain.talk("birds nest in trees");
    assert!(
        resp.contains("birds nest in trees"),
        "newborn echo missing for statement: {}",
        resp
    );
    assert!(!resp.contains("don't know"), "unexpected humble reply for a statement: {}", resp);

    let resp = brain.talk("what is a bird?");
    assert!(resp.contains("don't know"), "question echoed like a statement: {}", resp);
}

#[test]
fn end_bounds_records() {
    let mut brain = fresh_brain();
    let path = temp_file(
        "records.txt",
        "TYPE: math\nNAME: alpha\nDEFINITION: first letter\nEND\n\
         TYPE: math\nNAME: beta\nDEFINITION: second letter\nEND\n",
    );
    brain.teach_file(&path).unwrap();

    let resp = brain.talk("what is alpha");
    assert!(resp.contains("first letter"), "alpha definition wrong: {}", resp);
    let resp = brain.talk("what is beta");
    assert!(resp.contains("second letter"), "beta definition wrong: {}", resp);
}