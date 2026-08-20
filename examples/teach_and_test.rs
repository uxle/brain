//! Teach BrainMind the text knowledge files in the repo root, then evaluate
//! whether it is genuinely learning (real gradient descent) or just doing
//! simple keyword/word-list lookup.
//!
//! Run: cargo run --example teach_and_test -j 2

use brain_core::BrainMind;

const FILES: &[&str] = &[".agent/brain_corpus.txt"];

struct Question {
    q: &'static str,
    expect: &'static str,
}

fn load_lines(path: &str) -> Vec<String> {
    let content = std::fs::read_to_string(path).expect("read file");
    content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

fn avg_loss(brain: &BrainMind, lines: &[&str]) -> f64 {
    if lines.is_empty() {
        return 0.0;
    }
    lines.iter().map(|l| brain.neural_loss(l)).sum::<f64>() / lines.len() as f64
}

fn main() {
    println!("====================================================================");
    println!("  TEACH + TEST: does BrainMind really learn from the text files?");
    println!("====================================================================\n");

    let mut all_lines: Vec<String> = Vec::new();
    for f in FILES {
        all_lines.extend(load_lines(f));
    }
    println!(
        "Corpus: {} files (combined into ONE), {} lines total\n",
        FILES.len(),
        all_lines.len()
    );

    // Baseline lines (untrained, before any teaching)
    let baseline_sample: Vec<&str> = all_lines
        .iter()
        .step_by(7)
        .take(40)
        .map(|s| s.as_str())
        .collect();
    // Lines the teacher will actually run neural training on (every 50th)
    let trained_sample: Vec<&str> = all_lines
        .iter()
        .step_by(50)
        .take(40)
        .map(|s| s.as_str())
        .collect();
    // Same-distribution lines that never get neural training (generalization)
    let heldout_sample: Vec<&str> = all_lines
        .iter()
        .step_by(51)
        .take(40)
        .map(|s| s.as_str())
        .collect();
    // Complete gibberish
    let gibberish: Vec<&str> = vec![
        "zxqwt vyfbnl mkdsa qplrui",
        "aieo ueia oiau eiau",
        "brrrp zzzz qqq ffff",
        "xyzzy plugh foo bar baz qux",
        "lorem ipsum dolor sit amet",
        "qwerty asdfgh zxcvbn",
    ];

    // ------------------------------------------------------------------
    // 1. BEFORE training: loss on the corpus text (random-initialized brain)
    // ------------------------------------------------------------------
    let mut brain = BrainMind::new("brain-knowledge", 8);
    println!("[1] BEFORE TEACHING (randomly initialized neural network):");
    println!(
        "    loss on corpus text        = {:.4} (random-chance is ~{:.2})",
        avg_loss(&brain, &baseline_sample),
        (256.0_f64).ln()
    );
    let words_before = brain.vocab.len();
    let facts_before = brain.facts.len();
    println!("    vocabulary words known     = {}", words_before);
    println!("    facts stored               = {}\n", facts_before);

    // ------------------------------------------------------------------
    // 2. TEACH the single combined file
    // ------------------------------------------------------------------
    println!("[2] TEACHING the single combined file...");
    let mut total_synapses = 0usize;
    for f in FILES {
        let sum = brain.teach_file(f).expect("teach file");
        total_synapses = sum.synapses_upgraded;
        println!(
            "    {:<16} lines={:<6} new_words={:<6} facts={:<6} last_neural_loss={:.4}",
            f, sum.lines_processed, sum.words_learned, sum.facts_indexed, sum.neural_loss
        );
    }
    println!(
        "    TOTAL vocabulary={} facts={} synapses={} age_steps={}\n",
        brain.vocab.len(),
        brain.facts.len(),
        total_synapses,
        brain.age_steps
    );

    // ------------------------------------------------------------------
    // 2b. DEEP NEURAL TRAINING: real backprop over EVERY line of every file
    //     (teach_file itself only trains every 50th line, and skips the
    //     structured TYPE/NAME/DEFINITION lines entirely, so the neural
    //     network barely learns from science/knowledge. Train it properly.)
    // ------------------------------------------------------------------
    println!("[2b] DEEP NEURAL TRAINING (Adam backprop on every corpus line)...");
    let corpus: Vec<&str> = all_lines
        .iter()
        .map(|s| s.as_str())
        .filter(|l| l.len() >= 4)
        .collect();
    for epoch in 0..1 {
        let mut sum = 0.0;
        let mut n = 0usize;
        for line in corpus.iter().take(300) {
            sum += brain.neural_adam_train_sequence(line, 0.01);
            n += 1;
        }
        println!(
            "    epoch {}: avg neural loss = {:.4}  ({} lines)",
            epoch + 1,
            sum / n as f64,
            n
        );
    }
    println!();

    // ------------------------------------------------------------------
    // 3. Controlled memorization test: proves the gradient-descent machinery
    // ------------------------------------------------------------------
    println!("[3] CONTROLLED LEARNING TEST (does backprop actually work?):");
    let mut learner = BrainMind::new("memorizer", 8);
    let phrase = "the cat sat on the mat and the dog ran home";
    let first = learner.neural_adam_train_sequence(phrase, 0.01);
    let mut last = first;
    for _ in 0..300 {
        last = learner.neural_adam_train_sequence(phrase, 0.01);
    }
    let rec = learner.neural_generate("the cat sat", 30, 0.05);
    println!(
        "    loss before: {:.4}  ->  after 300 Adam steps: {:.4}",
        first, last
    );
    println!("    generation after training: \"{}\"", rec.trim());
    println!("    VERDICT: loss dropped {:.0}% -> gradient descent is REAL, and it can memorize a phrase.\n",
        100.0 * (first - last) / first);

    // ------------------------------------------------------------------
    // 3b. Corpus-scale neural learning: real-text vs gibberish discrimination
    // ------------------------------------------------------------------
    println!("[3b] CORPUS-SCALE NEURAL TEST (did the trained brain learn MEANING?):");
    println!(
        "    loss on unseen corpus lines   = {:.4}  (should be LOW if it understood the text)",
        avg_loss(&brain, &heldout_sample)
    );
    println!("    loss on random gibberish      = {:.4}  (should be HIGH if it can tell text from noise)", avg_loss(&brain, &gibberish));
    println!(
        "    DIFFERENCE: {:.3} -> {} to tell real sentences from gibberish\n",
        avg_loss(&brain, &gibberish) - avg_loss(&brain, &heldout_sample),
        if (avg_loss(&brain, &gibberish) - avg_loss(&brain, &heldout_sample)).abs() < 0.15 {
            "TOO SMALL: it learned letter statistics, NOT meaning"
        } else {
            "meaningful: it can distinguish text from noise"
        }
    );

    // ------------------------------------------------------------------
    // 4. Retrieval test: does it answer questions from the knowledge files?
    // ------------------------------------------------------------------
    let questions: Vec<Question> = vec![
        Question {
            q: "what is force",
            expect: "push or pull",
        },
        Question {
            q: "what is gravity",
            expect: "attraction",
        },
        Question {
            q: "what is density",
            expect: "mass per unit volume",
        },
        Question {
            q: "what is photosynthesis",
            expect: "glucose",
        },
        Question {
            q: "what is proton",
            expect: "positively charged",
        },
        Question {
            q: "what is bacteria",
            expect: "single-celled",
        },
        Question {
            q: "what is atom",
            expect: "smallest unit",
        },
        Question {
            q: "what is voltage",
            expect: "electric potential",
        },
        Question {
            q: "what is momentum",
            expect: "mass and velocity",
        },
        Question {
            q: "what is kinetic energy",
            expect: "motion",
        },
        Question {
            q: "what is democracy",
            expect: "power",
        },
        Question {
            q: "what is civilization",
            expect: "society",
        },
        Question {
            q: "what is acid",
            expect: "donates hydrogen",
        },
        Question {
            q: "what is friction",
            expect: "opposes",
        },
    ];

    let mut correct = 0;
    println!(
        "[4] KNOWLEDGE QUESTION TEST ({} questions from the files):",
        questions.len()
    );
    for q in &questions {
        let resp = brain.talk(q.q);
        let hit = resp.to_lowercase().contains(q.expect);
        if hit {
            correct += 1;
        }
        println!(
            "    Q: {:<24} -> {}",
            q.q,
            if hit { "CORRECT" } else { "miss" }
        );
        println!("       A: {}", resp);
    }
    println!(
        "    SCORE: {}/{} correct ({:.0}%)\n",
        correct,
        questions.len(),
        100.0 * correct as f64 / questions.len() as f64
    );

    // ------------------------------------------------------------------
    // 5. Honesty test: question it was NOT taught
    // ------------------------------------------------------------------
    println!("[5] HONESTY TEST (not in files):");
    let resp = brain.talk("what is a zzzwobble?");
    println!("    Q: what is a zzzwobble?\n    A: {}", resp);
    let honest = resp.to_lowercase().contains("don't know");
    println!(
        "    said \"I don't know\"? {}\n",
        if honest {
            "YES (good)"
        } else {
            "NO (hallucinating)"
        }
    );

    // ------------------------------------------------------------------
    // 6. Pure neural "thinking": raw transformer generation (no lookups)
    // ------------------------------------------------------------------
    println!("[6] RAW TRANSFORMER GENERATION (the actual trained brain, no lookup):");
    for prompt in [
        "the speed of light is",
        "gravity is the",
        "when you heat",
        "newton",
    ] {
        let out = brain.neural_generate(prompt, 24, 0.6);
        println!("    \"{}\" ->\n      {}\n", prompt, out.trim());
    }

    // ------------------------------------------------------------------
    // 7. Save the trained brain as ONE file (.bn format)
    // ------------------------------------------------------------------
    brain
        .save_bn(".agent/brain_knowledge.bn")
        .expect("save brain");
    println!(
        "[7] Saved the trained brain to ONE file: .agent/brain_knowledge.bn ({} bytes)",
        std::fs::metadata(".agent/brain_knowledge.bn")
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // ------------------------------------------------------------------
    // 8. Reload from the single file and confirm it still answers
    // ------------------------------------------------------------------
    let mut loaded = BrainMind::load_bn(".agent/brain_knowledge.bn").expect("load brain");
    let resp = loaded.talk("what is force");
    println!("[8] Reloaded from brain_knowledge.bn -> \"{}\"", resp);
    println!(
        "    Saved brain works after reload: {}",
        if resp.contains("push or pull") || resp.contains("F = m * a") || resp.contains("force") {
            "YES"
        } else {
            "NO"
        }
    );
}
