//! Converts a model checkpoint from NamedMpk format (.mpk) to binary format (.bin).
//!
//! The binary format is required for WASM deployment, where `BinBytesRecorder` is
//! used to load the model from `include_bytes!`.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin convert_model_to_bin -- <input_path_without_extension>
//! ```
//!
//! Example:
//! ```bash
//! cargo run --bin convert_model_to_bin -- data/v5
//! ```
//!
//! This reads `data/v5.mpk` (NamedMpk format) and writes `data/v5.bin` (binary format).

use burn::backend::NdArray;
use ml_model_training::{load_checkpoint, save_checkpoint_bin};

type ConversionBackend = NdArray;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("Usage: convert_model_to_bin <path_without_extension>"))?;

    println!("Loading model from {input_path}.mpk ...");
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = load_checkpoint::<ConversionBackend>(input_path, &device)?;

    println!("Saving model to {input_path}.bin ...");
    save_checkpoint_bin(&model, input_path)?;

    println!("Conversion complete: {input_path}.mpk -> {input_path}.bin");
    Ok(())
}
