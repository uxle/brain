//! # USB HID Actuator Abstraction & Serial Protocol
//!
//! Controls keyboard and mouse actions on target systems via USB HID emulation (e.g. Raspberry Pi Pico).

use std::sync::{Arc, Mutex};

/// Key modifiers for keyboard actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyModifier {
    Ctrl,
    Shift,
    Alt,
    Meta,
}

/// Keyboard action command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyAction {
    Press(u8),
    Release(u8),
    TypeStr(String),
    Combo(Vec<KeyModifier>, u8),
}

/// Mouse button identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Mouse action command.
#[derive(Debug, Clone, PartialEq)]
pub enum MouseAction {
    MoveRel {
        dx: i32,
        dy: i32,
    },
    MoveAbs {
        x: u32,
        y: u32,
    },
    Click(MouseButton),
    DoubleClick(MouseButton),
    MouseDown(MouseButton),
    MouseUp(MouseButton),
    Scroll {
        dy: i32,
    },
    Drag {
        from_x: u32,
        from_y: u32,
        to_x: u32,
        to_y: u32,
    },
}

/// Unified HID Action.
#[derive(Debug, Clone, PartialEq)]
pub enum HidAction {
    Key(KeyAction),
    Mouse(MouseAction),
    DelayMs(u32),
}

/// Abstract HID Actuator Interface.
pub trait HidDevice: Send + Sync {
    /// Dispatches a HID action to the physical or mock device.
    fn execute(&self, action: &HidAction) -> Result<(), String>;

    /// Convenience: Types a string directly.
    fn type_text(&self, text: &str) -> Result<(), String> {
        self.execute(&HidAction::Key(KeyAction::TypeStr(text.to_string())))
    }

    /// Convenience: Moves mouse to absolute coordinate.
    fn move_mouse(&self, x: u32, y: u32) -> Result<(), String> {
        self.execute(&HidAction::Mouse(MouseAction::MoveAbs { x, y }))
    }

    /// Convenience: Clicks left mouse button.
    fn click_left(&self) -> Result<(), String> {
        self.execute(&HidAction::Mouse(MouseAction::Click(MouseButton::Left)))
    }
}

/// In-Memory Mock HID device for testing and recording action logs.
#[derive(Debug, Clone, Default)]
pub struct MockHidDevice {
    pub history: Arc<Mutex<Vec<HidAction>>>,
}

impl MockHidDevice {
    pub fn new() -> Self {
        Self {
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn actions(&self) -> Vec<HidAction> {
        self.history.lock().unwrap().clone()
    }

    pub fn clear(&self) {
        self.history.lock().unwrap().clear();
    }
}

impl HidDevice for MockHidDevice {
    fn execute(&self, action: &HidAction) -> Result<(), String> {
        self.history.lock().unwrap().push(action.clone());
        Ok(())
    }
}

/// Serial Binary Protocol for USB HID Pico Dongle.
/// Packet format: [0xAA, 0x55, CMD_BYTE, PAYLOAD_LEN, ...PAYLOAD, CRC8]
pub struct SerialHidProtocol;

impl SerialHidProtocol {
    pub const HEADER_1: u8 = 0xAA;
    pub const HEADER_2: u8 = 0x55;

    pub const CMD_KEY_PRESS: u8 = 0x01;
    pub const CMD_KEY_RELEASE: u8 = 0x02;
    pub const CMD_KEY_TYPE: u8 = 0x03;
    pub const CMD_MOUSE_MOVE: u8 = 0x10;
    pub const CMD_MOUSE_CLICK: u8 = 0x11;

    /// Encodes a HID action into a framed serial byte packet.
    pub fn encode_action(action: &HidAction) -> Vec<u8> {
        let mut packet = vec![Self::HEADER_1, Self::HEADER_2];

        match action {
            HidAction::Key(KeyAction::Press(k)) => {
                packet.push(Self::CMD_KEY_PRESS);
                packet.push(1);
                packet.push(*k);
            }
            HidAction::Key(KeyAction::Release(k)) => {
                packet.push(Self::CMD_KEY_RELEASE);
                packet.push(1);
                packet.push(*k);
            }
            HidAction::Key(KeyAction::TypeStr(s)) => {
                let bytes = s.as_bytes();
                packet.push(Self::CMD_KEY_TYPE);
                packet.push(bytes.len().min(255) as u8);
                packet.extend_from_slice(&bytes[..bytes.len().min(255)]);
            }
            HidAction::Mouse(MouseAction::MoveAbs { x, y }) => {
                packet.push(Self::CMD_MOUSE_MOVE);
                packet.push(8);
                packet.extend_from_slice(&x.to_le_bytes());
                packet.extend_from_slice(&y.to_le_bytes());
            }
            HidAction::Mouse(MouseAction::Click(btn)) => {
                packet.push(Self::CMD_MOUSE_CLICK);
                packet.push(1);
                packet.push(match btn {
                    MouseButton::Left => 1,
                    MouseButton::Right => 2,
                    MouseButton::Middle => 3,
                });
            }
            _ => {
                packet.push(0x00);
                packet.push(0);
            }
        }

        // Compute CRC-8 checksum
        let mut crc = 0u8;
        for &b in &packet[2..] {
            crc = crc.wrapping_add(b);
        }
        packet.push(crc);

        packet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_hid_device() {
        let hid = MockHidDevice::new();
        hid.type_text("hello").unwrap();
        hid.move_mouse(100, 200).unwrap();
        hid.click_left().unwrap();

        let acts = hid.actions();
        assert_eq!(acts.len(), 3);
        assert_eq!(acts[0], HidAction::Key(KeyAction::TypeStr("hello".into())));
    }

    #[test]
    fn test_serial_protocol_encoding() {
        let act = HidAction::Key(KeyAction::TypeStr("hi".into()));
        let pkt = SerialHidProtocol::encode_action(&act);
        assert_eq!(pkt[0], 0xAA);
        assert_eq!(pkt[1], 0x55);
        assert_eq!(pkt[2], SerialHidProtocol::CMD_KEY_TYPE);
        assert_eq!(pkt[3], 2);
        assert_eq!(&pkt[4..6], b"hi");
    }
}
