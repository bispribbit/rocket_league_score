//! Root layout and route switch for [`crate::app_state::AppState`].

use dioxus::prelude::*;

use crate::app_state::AppState;
use crate::branding::SMURF_SUSPECT_BADGE;

use super::results::{ErrorPage, UnsupportedReplayPage};
use super::upload::UploadPage;

/// Root application component.
#[component]
#[expect(clippy::volatile_composites)] // from dioxus asset! macro
pub(crate) fn App() -> Element {
    let state = use_signal(|| AppState::WaitingForUpload);

    rsx! {
        document::Stylesheet { href: asset!("/assets/tailwind.css") }
        document::Link {
            rel: "icon",
            r#type: Some("image/png".to_string()),
            href: Some(SMURF_SUSPECT_BADGE.into()),
        }

        div { class: "min-h-screen bg-gray-950 text-gray-100",
            match state() {
                AppState::WaitingForUpload => rsx! {
                    UploadPage { state }
                },
                AppState::Error(message) => rsx! {
                    ErrorPage { message, state }
                },
                AppState::UnsupportedReplay(details) => rsx! {
                    UnsupportedReplayPage { details, state }
                },
            }
        }
    }
}
