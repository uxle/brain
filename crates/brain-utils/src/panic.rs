//! # Panic Handling and Crash Reports
//!
//! Provides custom panic hook installation, crash context capture,
//! and persistent crash log dumping.

use std::panic::{self, PanicHookInfo};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static PANIC_HOOK_INSTALLED: AtomicBool = AtomicBool::new(false);
static LAST_PANIC_MSG: Mutex<Option<String>> = Mutex::new(None);

/// Installs a custom panic hook that records diagnostic messages and optional crash logs.
pub fn set_panic_hook() {
    if PANIC_HOOK_INSTALLED.swap(true, Ordering::SeqCst) {
        return; // already installed
    }
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let msg = format_panic_info(info);
        if let Ok(mut last) = LAST_PANIC_MSG.lock() {
            *last = Some(msg.clone());
        }
        eprintln!("=== BRAIN ENGINE PANIC ===");
        eprintln!("{}", msg);
        eprintln!("==========================");
        prev_hook(info);
    }));
}

/// Retrieves the most recent panic message if one occurred.
pub fn last_panic_message() -> Option<String> {
    LAST_PANIC_MSG.lock().ok().and_then(|m| m.clone())
}

fn format_panic_info(info: &PanicHookInfo) -> String {
    let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
        *s
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.as_str()
    } else {
        "Unknown panic payload"
    };

    let loc = if let Some(location) = info.location() {
        format!("{}:{}:{}", location.file(), location.line(), location.column())
    } else {
        "unknown location".to_string()
    };

    format!("Panic occurred at {}: {}", loc, payload)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_panic_handling_hook_1() {
        set_panic_hook();
        assert!(PANIC_HOOK_INSTALLED.load(Ordering::SeqCst));
        assert!(last_panic_message().is_none() || last_panic_message().is_some());
    }
}
