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
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

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
