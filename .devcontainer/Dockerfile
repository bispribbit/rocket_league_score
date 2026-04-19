# Bookworm matches NVIDIA's CUDA apt repo (debian12). Using trixie + that repo
# mixes libc/toolchain versions and apt typically fails installing cuda-* (exit 100).
FROM rust:1.92.0-bookworm

# Install additional tools for Rust development + Vulkan for GPU compute
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    llvm \
    clang \
    lld \
    lldb \
    wget \
    gnupg \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libpq-dev \
    postgresql-client \
    # Vulkan libraries for GPU compute (wgpu/burn)
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# CUDA 12.6 toolkit for burn-cuda / cubecl-cuda backend.
#
# We only install the bits cubecl-cuda actually needs at build/runtime:
#   - nvcc   : offline kernel compilation fallback
#   - cudart : CUDA runtime API
#   - nvrtc  : runtime kernel compilation (primary path)
#   - cublas : matmul kernels used inside LSTM/linear layers
#
# The host NVIDIA driver (libcuda.so) is mounted in from WSL at runtime,
# we do NOT install any nvidia-driver packages in the container.
#
# NVIDIA ships this repo for Debian 12 (bookworm) only; the image base must
# be bookworm so apt can satisfy cuda-* dependencies.
# ---------------------------------------------------------------------------
ARG CUDA_APT_VERSION=12-6
ARG CUDA_INSTALL_DIR=/usr/local/cuda-12.6
RUN wget -qO /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_APT_VERSION} \
        cuda-cudart-dev-${CUDA_APT_VERSION} \
        cuda-nvrtc-dev-${CUDA_APT_VERSION} \
        libcublas-dev-${CUDA_APT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=${CUDA_INSTALL_DIR}
ENV CUDA_PATH=${CUDA_INSTALL_DIR}
# Overrides `.cargo/config.toml` CUDARC_CUDA_VERSION for this image (Cargo does not let
# `[env]` replace an already-set variable). Keeps cudarc bindings aligned with CUDA 12.6.
ENV CUDARC_CUDA_VERSION=12060
ENV PATH=${CUDA_INSTALL_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
# LibTorch 2.4.0 + CUDA 12.1 — for tch-rs (cuDNN-fused LSTM in training).
#
# PyTorch's CUDA 12.1 build runs on any 12.x driver (our driver is 12.7).
# We use the CXX11-ABI build so it links against the system libstdc++.
# tch-rs (0.17.0) targets PyTorch / LibTorch 2.4.0.
#
# The LIBTORCH env var tells tch-rs's build script where to find headers +
# libraries. LD_LIBRARY_PATH ensures the runtime linker finds libtorch.so.
# ---------------------------------------------------------------------------
ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu121.zip
ARG LIBTORCH_DIR=/opt/libtorch
RUN wget -qO /tmp/libtorch.zip "${LIBTORCH_URL}" \
    && unzip -q /tmp/libtorch.zip -d /opt/ \
    && rm /tmp/libtorch.zip

ENV LIBTORCH=${LIBTORCH_DIR}
ENV LD_LIBRARY_PATH=${LIBTORCH_DIR}/lib:${LD_LIBRARY_PATH}

# Tailwind CSS CLI (matches crates/is_this_a_smurf/package.json)
RUN npm install -g @tailwindcss/cli@4.1.3

# Install rust components and WebAssembly target (e.g. Dioxus web)
# Nightly rustfmt is used for rustfmt.toml unstable options (cargo make fmt).
RUN rustup component add rustfmt clippy rust-src llvm-tools \
    && rustup target add wasm32-unknown-unknown \
    && rustup toolchain install nightly \
    && rustup component add --toolchain nightly-x86_64-unknown-linux-gnu rustfmt

# Install cargo-binstall
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Install cargo tools using cargo-binstall
RUN cargo binstall cargo-make cargo-edit sqlx-cli dioxus-cli -y

# Set the working directory
WORKDIR /workspace

# Keep container running
CMD ["/bin/bash"]
