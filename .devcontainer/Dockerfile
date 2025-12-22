FROM rust:1.92.0-trixie

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
    pkg-config \
    libssl-dev \
    libpq-dev \
    postgresql-client \
    # Vulkan libraries for GPU compute (wgpu/burn)
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Install rust components
RUN rustup component add rustfmt clippy rust-src llvm-tools

# Install cargo-binstall
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Install cargo tools using cargo-binstall
RUN cargo binstall cargo-make cargo-edit sqlx-cli -y

# Set the working directory
WORKDIR /workspace

# Keep container running
CMD ["/bin/bash"]
