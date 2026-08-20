//! # Hardware Abstraction Layer (HAL)
//!
//! Sensor and actuator interfaces for autonomous physical and virtual agent interaction.

pub mod hid;
pub mod video;
pub mod audio;
pub mod safety;

pub use hid::{HidAction, HidDevice, KeyAction, KeyModifier, MockHidDevice, MouseAction, MouseButton, SerialHidProtocol};
pub use video::{MockVideoSource, VideoFrame, VideoSource};
pub use audio::{AudioChunk, AudioSource, MockAudioSource};
pub use safety::{SafetyConfig, SafetyGuard};
