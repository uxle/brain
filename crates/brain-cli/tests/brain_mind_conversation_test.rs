//! # Dynamic Brain Mind Real-Time Learning & Conversational Tests

use brain_core::brain_mind::BrainMind;

#[test]
fn test_newborn_baby_imitation_and_growth() {
    let mut brain = BrainMind::new("baby_brain", 10);
    assert_eq!(brain.total_neurons(), 1000);
    assert_eq!(brain.age_steps, 0);

    // Turn 1 (Newborn baby stage): echoes greeting
    let resp1 = brain.talk("hi");
    assert!(resp1.contains("hi"), "Newborn brain must mimic greeting: got '{}'", resp1);
    assert_eq!(brain.age_steps, 1);

    // Turn 2: learns nickname
    let resp2 = brain.talk("my name is Lion");
    assert!(resp2.contains("Lion"), "Brain must recognize user's name: got '{}'", resp2);

    // Turn 3: recalls user's identity
    let resp3 = brain.talk("who am I?");
    assert!(resp3.contains("Lion"), "Brain must recall teacher's name: got '{}'", resp3);

    // Turn 4: basic mathematics
    let resp4 = brain.talk("what is 2 + 2");
    assert_eq!(resp4, "2 + 2 = 4");

    let resp5 = brain.talk("what is 10 * 5");
    assert_eq!(resp5, "10 * 5 = 50");
}

#[test]
fn test_brain_teaching_from_text_and_memory_persistence() {
    let mut brain = BrainMind::new("scholar_brain", 5);

    // Teach grammar knowledge
    brain.learn_sentence("A noun is a word that represents a person, place, or thing.");
    brain.learn_sentence("A verb is an action word.");
    brain.learn_sentence("Mathematics is the language of science.");

    assert!(brain.vocab.len() > 10);
    assert!(brain.facts.len() >= 3);

    // Query facts
    let resp_noun = brain.talk("noun");
    assert!(resp_noun.contains("person, place, or thing") || resp_noun.contains("noun is"));

    // Save to .bn binary file
    let bn_file = "target/scholar_test.bn";
    let _ = std::fs::remove_file(bn_file);

    brain.save_bn(bn_file).expect("Save .bn");

    // Reload brain from .bn file
    let loaded = BrainMind::load_bn(bn_file).expect("Load .bn");
    assert_eq!(loaded.name, "scholar_brain");
    assert_eq!(loaded.cube_dim, 5);
    assert_eq!(loaded.total_neurons(), 125);
    assert_eq!(loaded.vocab.len(), brain.vocab.len());
    assert_eq!(loaded.facts.len(), brain.facts.len());

    let _ = std::fs::remove_file(bn_file);
}

#[test]
fn test_teaching_corpus_files_data_and_math() {
    let mut brain = BrainMind::new("educated_brain", 10);

    // If data.txt exists in root, teach first 50 lines
    if std::path::Path::new("data.txt").exists() {
        let stats = brain.teach_file("data.txt").expect("Teach data.txt");
        assert!(stats.lines_processed > 100);
        assert!(stats.words_learned > 100);
    }
}
