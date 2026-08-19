//! # Global Initialization & Lifecycle Execution
//!
//! Provides top-level initialization, shutdown orchestration, and runtime summaries.

use std::sync::{Arc, Mutex};
use crate::core::{GlobalState, UtilsConfig, UtilsResult};
use crate::panic::set_panic_hook;

static GLOBAL_STATE: Mutex<Option<Arc<GlobalState>>> = Mutex::new(None);

/// Initializes global framework utilities (panic hooks, logging, profiling).
pub fn init_utils(config: UtilsConfig) -> UtilsResult<Arc<GlobalState>> {
    set_panic_hook();
    let state = Arc::new(GlobalState::new(config));
    let mut g = GLOBAL_STATE.lock().unwrap();
    *g = Some(state.clone());
    Ok(state)
}

/// Retrieves the active global runtime state if initialized.
pub fn get_global_state() -> Option<Arc<GlobalState>> {
    GLOBAL_STATE.lock().ok().and_then(|g| g.clone())
}

/// Shuts down utilities and flushes log sinks.
pub fn shutdown_utils() {
    let mut g = GLOBAL_STATE.lock().unwrap();
    *g = None;
}

/// Generates a diagnostic summary string of active utility services.
pub fn utils_summary() -> String {
    if let Some(state) = get_global_state() {
        let cfg = state.get_config();
        format!(
            "Brain Utils [App: {}, Level: {}, Profiling: {}, Uptime: {:?}]",
            cfg.app_name, cfg.log_level, cfg.profiling_enabled, state.uptime()
        )
    } else {
        "Brain Utils [Uninitialized]".to_string()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_impl_lifecycle_and_summary_1() {
        let cfg = UtilsConfig::default().with_app_name("test_suite_1");
        let state = init_utils(cfg).unwrap();
        assert!(state.is_initialized());
        let _ = state.uptime();
    
        let summary = utils_summary();
        assert!(summary.starts_with("Brain Utils"));
    }
}
