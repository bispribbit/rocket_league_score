use burn::backend::cuda::CudaDevice;
use tracing::info;

/// Initializes a CUDA device for GPU acceleration.
///
/// Requires the CUDA toolkit (see `.devcontainer/Dockerfile`) and an NVIDIA
/// GPU visible to the container.
///
/// This function only exists so the backend device can be swapped in a single
/// place if we ever need to change it (e.g. for a CPU-only CI runner).
pub fn init_device() -> CudaDevice {
    info!("Initializing CUDA device...");
    CudaDevice::default()
}
