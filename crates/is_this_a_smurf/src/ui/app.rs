//! Root layout and route switch for [`crate::app_state::AppState`].

use dioxus::prelude::*;

use crate::app_state::{AppState, ResultsScreenState};

use super::results::{ErrorPage, ResultsPage};
use super::upload::UploadPage;

/// Root application component.
#[component]
#[expect(clippy::volatile_composites)] // from dioxus asset! macro
pub(crate) fn App() -> Element {
    let state = use_signal(|| AppState::WaitingForUpload);

    rsx! {
        document::Stylesheet { href: asset!("/assets/tailwind.css") }

        div { class: "min-h-screen bg-gray-950 text-gray-100",
            match state() {
                AppState::WaitingForUpload => rsx! {
                    UploadPage { state }
                },
                AppState::ShowingResults(screen) => {
                    let ResultsScreenState {
                        filename,
                        results,
                        timeline_progress,
                    } = *screen;
                    rsx! {
                        ResultsPage {
                            filename,
                            results,
                            timeline_progress,
                            state,
                        }
                    }
                },
                AppState::Error(message) => rsx! {
                    ErrorPage { message, state }
                },
            }
        }
    }
}
