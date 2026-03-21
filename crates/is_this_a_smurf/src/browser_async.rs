//! Small async helpers for the browser event loop (WASM).

use dioxus::prelude::document;

/// Yields to the browser event loop so the UI can repaint.
#[expect(clippy::future_not_send, clippy::let_underscore_untyped)]
pub(crate) async fn yield_to_ui() {
    let _ = document::eval("new Promise(r => setTimeout(r, 0))").await;
}

/// Waits for UI animations between timeline phases (WASM).
#[expect(clippy::future_not_send, clippy::let_underscore_untyped)]
pub(crate) async fn sleep_milliseconds(milliseconds: u32) {
    let script = format!("new Promise(r => setTimeout(r, {milliseconds}))");
    let _ = document::eval(&script).await;
}
