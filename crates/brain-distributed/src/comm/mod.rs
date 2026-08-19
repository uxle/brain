//! # Inter-Process Communication Layer
//!
//! Provides [`CommBackend`], in-memory channels (`MemComm`), and TCP sockets (`TcpComm`).

pub mod message;
pub mod serialize;

pub use message::MessageHeader;
pub use serialize::serialize_tensor;

/// Abstract communication transport interface.
pub trait CommBackend: Send + Sync {
    fn send_bytes(&self, dest: usize, data: &[u8]) -> Result<(), String>;
    fn recv_bytes(&self, src: usize) -> Result<Vec<u8>, String>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
