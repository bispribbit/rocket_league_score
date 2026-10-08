use burn::prelude::{Device, DeviceKind};
use tracing::{info, warn};

/// Environment variable: if set to a decimal `u64`, seeds the device so GPU-side
/// randomness (e.g. label jitter, dropout masks) matches a chosen run. The `overfit_wgpu`
/// harness seeds with `42` by default; set `ROCKET_LEAGUE_WGPU_SEED=42` here for similar
/// reproducibility during full training.
pub const WGPU_SEED_ENV: &str = "ROCKET_LEAGUE_WGPU_SEED";

/// Initializes a wgpu device for GPU acceleration.
///
/// Works on Windows (DX12/Vulkan), Linux (Vulkan), and macOS (Metal) without
/// any extra runtime installation. Pass a more specific [`DeviceKind`] if you
/// need to pin a particular adapter.
///
/// Gradient recording is **not** enabled here: callers that train add it with
/// [`Device::autodiff`], and inference paths get a plain device.
pub fn init_device() -> Device {
    info!("Initializing wgpu device...");
    let device = Device::wgpu(DeviceKind::default());
    match std::env::var(WGPU_SEED_ENV) {
        Ok(seed_string) => match seed_string.parse::<u64>() {
            Ok(seed) => {
                device.seed(seed);
                info!(
                    seed,
                    "Seeded device PRNG from {WGPU_SEED_ENV} (overfit harness uses 42 by default)"
                );
            }
            Err(e) => warn!(
                variable = WGPU_SEED_ENV,
                value = %seed_string,
                error = %e,
                "Ignored invalid device seed"
            ),
        },
        Err(std::env::VarError::NotPresent) => {}
        Err(e) => warn!(error = %e, "Could not read {WGPU_SEED_ENV}"),
    }
    device
}
