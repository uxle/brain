//! # Hardware Abstraction Layer (HAL)
//!
//! Sensor and actuator interfaces for autonomous physical and virtual agent interaction.

pub mod audio;
pub mod hid;
pub mod safety;
pub mod video;

pub use audio::{AudioChunk, AudioSource, MockAudioSource};
pub use hid::{
    HidAction, HidDevice, KeyAction, KeyModifier, MockHidDevice, MouseAction, MouseButton,
    SerialHidProtocol,
};
pub use safety::{SafetyConfig, SafetyGuard};
pub use video::{MockVideoSource, VideoFrame, VideoSource};
