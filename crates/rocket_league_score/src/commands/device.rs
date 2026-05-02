use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use tracing::info;
use tracing::warn;

/// Environment variable: if set to a decimal `u64`, calls [`Wgpu::<f32>::seed`] so GPU-side
/// randomness (e.g. label jitter, dropout masks) matches a chosen run. The `overfit_wgpu`
/// harness seeds with `42` by default; set `ROCKET_LEAGUE_WGPU_SEED=42` here for similar
/// reproducibility during full training.
pub const WGPU_SEED_ENV: &str = "ROCKET_LEAGUE_WGPU_SEED";

/// Initializes a Wgpu device for GPU acceleration.
///
/// Works on Windows (DX12/Vulkan), Linux (Vulkan), and macOS (Metal) without
/// any extra runtime installation. Swap `WgpuDevice::default()` for
/// `WgpuDevice::DiscreteGpu(0)` if you need to pin a specific adapter.
pub fn init_device() -> WgpuDevice {
    info!("Initializing Wgpu device...");
    let device = WgpuDevice::default();
    match std::env::var(WGPU_SEED_ENV) {
        Ok(seed_string) => match seed_string.parse::<u64>() {
            Ok(seed) => {
                Wgpu::<f32>::seed(&device, seed);
                info!(
                    seed,
                    "Seeded Wgpu PRNG from {WGPU_SEED_ENV} (overfit harness uses 42 by default)"
                );
            }
            Err(e) => warn!(
                variable = WGPU_SEED_ENV,
                value = %seed_string,
                error = %e,
                "Ignored invalid Wgpu seed"
            ),
        },
        Err(std::env::VarError::NotPresent) => {}
        Err(e) => warn!(error = %e, "Could not read {WGPU_SEED_ENV}"),
    }
    device
}
