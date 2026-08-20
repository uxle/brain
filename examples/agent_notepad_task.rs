//! # Autonomous Agent GUI Task Example ("Open Notepad & Type Hello")
//!
//! Demonstrates the Brain Autonomous Agent interacting with a virtual desktop via
//! simulated video capture and USB HID emulation.

use brain_core::Tensor;
use brain_rl::{IntrinsicCuriosityModule, Skill, SkillLibrary, SkillStep, WorldModel};
use brain_utils::hal::{
    HidAction, HidDevice, KeyAction, MockHidDevice, MockVideoSource, MouseAction, MouseButton,
    SafetyConfig, SafetyGuard, VideoSource,
};

fn main() {
    println!("============================================================");
    println!("  Brain Autonomous Agent — GUI Task: 'Open Notepad & Type'  ");
    println!("============================================================");

    // 1. Initialize Hardware Abstraction Layer (HAL)
    let video_source = MockVideoSource::new(1920, 1080);
    let hid_device = MockHidDevice::new();
    let safety_guard = SafetyGuard::new(SafetyConfig::default());

    // 2. Initialize Cognitive & Reinforcement Learning Models
    let state_dim = 16;
    let action_dim = 4; // [target_x, target_y, click_flag, key_flag]
    let world_model = WorldModel::new(state_dim, action_dim, 32);
    let curiosity = IntrinsicCuriosityModule::new(state_dim, action_dim, 16, 0.1);
    let mut skill_library = SkillLibrary::new();

    // 3. Register Pretrained / Prior Knowledge Skill: "type_greeting"
    let greeting_skill = Skill {
        name: "type_greeting".into(),
        goal_embedding: vec![0.5; state_dim],
        steps: vec![
            SkillStep {
                state_embedding: vec![0.5; state_dim],
                action: vec![500.0, 300.0, 1.0, 0.0], // Click text editor at (500, 300)
                reward: 1.0,
                description: Some("Focus text editor input area".into()),
            },
            SkillStep {
                state_embedding: vec![0.5; state_dim],
                action: vec![0.0, 0.0, 0.0, 1.0], // Type text
                reward: 2.0,
                description: Some("Type message".into()),
            },
        ],
        cumulative_reward: 3.0,
        success_count: 5,
    };
    skill_library.insert(greeting_skill);

    println!("[*] Initialized HAL, World Model, Curiosity, and Skill Library.");
    println!("[*] Step 1: Perceiving desktop screen from HDMI input...");

    let frame = video_source.capture_frame().expect("Video capture ok");
    let frame_tensor = frame.to_tensor();
    println!(
        "    [✓] Captured screen frame tensor with shape: {:?}",
        frame_tensor.shape()
    );

    // Compute coarse state representation from visual features
    let mut state_vec = Vec::with_capacity(state_dim);
    let data = frame_tensor.data();
    let chunk_size = data.len() / state_dim;
    for i in 0..state_dim {
        let chunk = &data[i * chunk_size..(i + 1) * chunk_size];
        state_vec.push(chunk.iter().sum::<f64>() / chunk_size as f64);
    }
    let state_tensor = Tensor::from_vec(state_vec.clone(), vec![state_dim]);

    println!("[*] Step 2: Querying Skill Library for matching action plan...");
    if let Some(skill) = skill_library.search(&state_vec, 0.5) {
        println!(
            "    [✓] Found high-confidence skill: '{}' (Historical success: {})",
            skill.name, skill.success_count
        );

        for (idx, step) in skill.steps.iter().enumerate() {
            println!(
                "    [Sub-Goal {}] {}",
                idx + 1,
                step.description.as_deref().unwrap_or("Action")
            );

            let hid_action = if idx == 0 {
                // Move mouse and click
                let x = step.action[0] as u32;
                let y = step.action[1] as u32;
                HidAction::Mouse(MouseAction::Drag {
                    from_x: 0,
                    from_y: 0,
                    to_x: x,
                    to_y: y,
                })
            } else {
                // Type greeting text
                HidAction::Key(KeyAction::TypeStr(
                    "Hello, Brain Autonomous Agent World!".into(),
                ))
            };

            // Verify safety guardrails
            safety_guard
                .verify_action(&hid_action)
                .expect("Action verified safe");

            // Execute on HID device
            hid_device.execute(&hid_action).expect("HID execution ok");
        }
    }

    println!("[*] Step 3: Verifying executed HID command history...");
    let history = hid_device.actions();
    for (i, act) in history.iter().enumerate() {
        println!("    Action #{}: {:?}", i + 1, act);
    }

    println!("[*] Step 4: Computing self-supervised prediction and curiosity reward...");
    let dummy_action = Tensor::from_slice(&[500.0, 300.0, 1.0, 0.0], vec![action_dim]);
    let next_frame = video_source.capture_frame().expect("Next frame capture ok");
    let next_frame_tensor = next_frame.to_tensor();
    let next_state_tensor = Tensor::from_vec(vec![0.52; state_dim], vec![state_dim]);

    let pred = world_model
        .predict(&state_tensor, &dummy_action)
        .expect("Prediction ok");
    let intrinsic_r = curiosity
        .compute_intrinsic_reward(&state_tensor, &dummy_action, &next_state_tensor)
        .unwrap_or(0.0);

    println!(
        "    [✓] World Model predicted next state error: {:.6}",
        (pred.next_state.data()[0] - 0.52).abs()
    );
    println!(
        "    [✓] Intrinsic Curiosity Exploration Reward: {:.4}",
        intrinsic_r
    );

    println!("============================================================");
    println!("  Task Completed: Agent autonomously actuated GUI target!   ");
    println!("============================================================");
}
