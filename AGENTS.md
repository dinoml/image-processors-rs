# Agent Instructions

## Scope

These instructions apply to the entire repository.

This is a Rust-first workspace for media loading, decoded frame ownership, tensor/layout conversion, and image processor primitives. Keep bindings and non-Rust integrations out of tree unless the task explicitly asks for them.

## Project Layout

- `Cargo.toml` defines the workspace.
- `image-processors/` contains the Rust library crate.
- `image-processors/src/media.rs` owns media source loading, image sequence decoding, optional URL loading, optional video decoding, and frame sampling.
- `image-processors/src/tensor.rs` owns tensor data, dtypes, shapes, and layouts.
- `image-processors/src/image.rs` owns processor config/options and image/video preprocessing entrypoints.
- `image-processors/src/transforms.rs` owns resize, crop, fill, and geometry helpers.
- `docs/FFMPEG.md` documents FFmpeg development-library setup for the optional `video` feature.
- `docs/ROADMAP.md` tracks current implementation and next milestones.

## Rust Conventions

- Prefer clear, idiomatic Rust over clever abstractions.
- Follow Microsoft's Pragmatic Rust Guidelines by default, unless these project instructions or an explicit user request require a different choice.
- Keep public APIs Rust-shaped: strong types, explicit ownership, and meaningful `Result`/`Option` boundaries.
- Avoid cloning decoded image/video buffers unless ownership requires it and the cost is understood.
- Keep feature-gated APIs coherent for `default`, `url`, `video`, and `url + video` builds.
- Use `thiserror` for library errors and preserve useful context in error variants.
- Do not use `unwrap()` in library code paths. In tests, prefer explicit assertions or `expect()` messages that explain the invariant.
- Document public APIs with rustdoc, including `# Errors`, `# Panics`, and `# Safety` sections when applicable.

## Testing And Verification

Run the narrowest useful command first, then broaden before finishing:

```powershell
cargo fmt --all --check
cargo test --workspace
cargo test --workspace --features url
```

Run Clippy when touching shared APIs, feature gates, media loading, tensor layout logic, or error handling:

```powershell
cargo clippy --workspace --all-targets -- -D warnings
```

The `video` feature requires FFmpeg development libraries, not just `ffmpeg.exe`. On Windows, use:

```powershell
.\scripts\setup-vcpkg-ffmpeg.ps1 -CheckCargo
cargo test --workspace --all-features
```

If the environment is missing FFmpeg development libraries, do not treat all-feature video failures as ordinary Rust failures. Report the missing native dependency and still run the default and `url` checks.

## Change Discipline

- Keep changes focused on the requested behavior.
- Preserve existing feature-gate behavior and add tests for both enabled and disabled feature paths when changing optional functionality.
- Update `README.md` or `docs/` when behavior, setup, or supported media formats change.
- Do not commit generated build outputs, local virtual environments, vcpkg installations, or `target/`.
