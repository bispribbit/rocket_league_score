# Rust coding convention
- Do not use abbreviation
- Use "Cargo clippy --all-features --all-targets" to check for compilation instead of "Cargo check"
    - Use Cargo clippy --all-features --all-targets --fix --allow-dirty for easy fixing of clippy lints when possible (Less token used & relialible)
- Cargo.toml : Only the workspace cargo.toml can have revision numbers, in the crates, always use the format crate_name.workspace = true
- Do not use Tuple, always create a named struct instead

# Sqlx
Always use the query! and query_as! macros instead of query, query_as unelss you absolutely must use a QueryBuilder.
Run sqlx migrate run in crates/database when you generate a new schema before trying to compile