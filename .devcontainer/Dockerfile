FROM ubuntu:24.04

# Why Ubuntu 24.04 (noble): the Kisak-mesa PPA packages a modern Mesa with the
# `dzn` Vulkan-on-D3D12 driver (`libvulkan_dzn.so`), which is what lets us
# reach the host NVIDIA GPU via WSL2's /dev/dxg. Debian's mesa-vulkan-drivers
# explicitly excludes `dzn` from its build, and building Mesa from source adds
# 20+ minutes to the container build. The PPA gives us a prebuilt binary.
#
# See:
#   https://gist.github.com/lzlrd/0d9b757d2122f551794fd00c7f0752d6
#   https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md

ENV DEBIAN_FRONTEND=noninteractive

# Core build tools + Vulkan loader + X11 dev headers (pulled in by Mesa at
# apt-install time for the dzn driver runtime deps).
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    llvm \
    clang \
    lld \
    lldb \
    wget \
    unzip \
    gnupg \
    ca-certificates \
    software-properties-common \
    pkg-config \
    libssl-dev \
    libpq-dev \
    postgresql-client \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Kisak-mesa PPA gives us Mesa 25+ with dzn; stock Ubuntu 24.04 ships 24.x
# without dzn. The `mesa-vulkan-drivers` package from this PPA drops both
# `libvulkan_dzn.so` and `/usr/share/vulkan/icd.d/dzn_icd.x86_64.json`.
RUN add-apt-repository -y ppa:kisak/kisak-mesa \
    && apt-get update \
    && apt-get install -y mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20+ (@tailwindcss/oxide requires >= 20).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Rust via rustup, pinned to 1.95.0. We cannot use the official `rust:*-noble`
# image as our base because we need the Kisak PPA available BEFORE installing
# mesa-vulkan-drivers, and the ordering of the base + PPA layers matters for
# caching. Install paths match the official Rust image so the existing
# `/usr/local/cargo/registry` volume mount still works.
# ---------------------------------------------------------------------------
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/rustup/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.95.0 --no-modify-path \
    && chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# Tailwind CSS CLI (matches crates/is_this_a_smurf/package.json).
RUN npm install -g @tailwindcss/cli@4.1.3

# Rust components + wasm target + nightly rustfmt (for unstable fmt options).
RUN rustup component add rustfmt clippy rust-src llvm-tools \
    && rustup target add wasm32-unknown-unknown \
    && rustup toolchain install nightly \
    && rustup component add --toolchain nightly-x86_64-unknown-linux-gnu rustfmt

RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

RUN cargo binstall cargo-make cargo-edit sqlx-cli dioxus-cli -y

WORKDIR /workspace

CMD ["/bin/bash"]
