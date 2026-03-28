//! Async helpers that truly yield to the browser event loop (WASM).
//!
//! **Why not `dioxus::prelude::document::eval`?**
//!
//! `document::eval("new Promise(…)").await` resolves when the JS expression is *evaluated*
//! (returning the Promise object), **not** when the Promise itself *resolves*.  That means
//! the previous `yield_to_ui` was effectively a no-op — synchronous Rust work immediately
//! resumed, blocking the main thread before the browser could repaint.
//!
//! Using `js_sys::eval` + `JsFuture::from(promise).await` genuinely suspends the Rust future
//! until the JS Promise resolves (i.e. after the `setTimeout` / `requestAnimationFrame` fires),
//! giving the browser a real opportunity to commit DOM changes and paint.

use js_sys::Promise;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

fn eval_promise(script: &str) -> Promise {
    js_sys::eval(script)
        .expect("JS eval failed for browser async helper")
        .unchecked_into()
}

/// Yields to the browser event loop so the UI can repaint.
#[expect(clippy::future_not_send)]
pub(crate) async fn yield_to_ui() {
    let _: Result<_, _> = JsFuture::from(eval_promise("new Promise(r => setTimeout(r, 0))")).await;
}

/// Waits until after the next paint so pipeline labels and spinners actually appear on screen.
///
/// Double `requestAnimationFrame` ensures the browser has
/// (a) calculated layout/style, and (b) painted, before the `setTimeout(0)` fires and our Rust
/// future resumes with the next chunk of synchronous work.
#[expect(clippy::future_not_send)]
pub(crate) async fn yield_for_dom_paint() {
    let _: Result<_, _> = JsFuture::from(eval_promise(
        "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 0))))",
    ))
    .await;
}

/// Sleeps for the given number of milliseconds (for timeline animations, etc.).
#[expect(clippy::future_not_send)]
pub(crate) async fn sleep_milliseconds(milliseconds: u32) {
    let script = format!("new Promise(r => setTimeout(r, {milliseconds}))");
    let _: Result<_, _> = JsFuture::from(eval_promise(&script)).await;
}
