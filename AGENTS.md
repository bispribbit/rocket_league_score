# Rust coding convention
- Do not use abbreviation
- Use "Cargo clippy --all-features --all-targets" to check for compilation instead of "Cargo check"
    - Use Cargo clippy --all-features --all-targets --fix --allow-dirty for easy fixing of clippy lints when possible (Less token used & relialible)
- Cargo.toml : Only the workspace cargo.toml can have revision numbers, in the crates, always use the format crate_name.workspace = true