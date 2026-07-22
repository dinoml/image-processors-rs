---
name: microsoft-rust-guidelines
description: Use when writing, reviewing, or refactoring Rust in this repository. Applies Microsoft's Pragmatic Rust Guidelines as the base Rust design and quality reference, including AI-agent-specific guidance, API design, correctness, performance, documentation, project structure, and safety rules.
---

# Microsoft Pragmatic Rust Guidelines

Use this skill as the base Rust guidance for this repository.

The bundled reference is a local copy of Microsoft's `agents/all.txt` export:

- `references/pragmatic-rust-guidelines-all.txt`
- Source: https://microsoft.github.io/rust-guidelines/agents/all.txt
- Downloaded: 2026-07-06
- SHA-256: `BF02C689591B1E46613DC3A5B19EDD53F1C68EEE39834E7ABBED6CCC984EC076`

## How To Use

1. Read the relevant sections of `references/pragmatic-rust-guidelines-all.txt` before making or reviewing Rust changes.
2. Treat the guidelines as the default baseline, especially the AI, correctness, library UX, performance, documentation, and project sections.
3. Apply the spirit of each guideline. If a guideline conflicts with repository-specific instructions in `AGENTS.md`, follow `AGENTS.md`.
4. Preserve idiomatic Rust API shape: strong types, explicit ownership, clear error boundaries, and testable APIs.
5. Verify changes with the commands in the repository `AGENTS.md`.

## Project Emphasis

For this image processor crate, pay extra attention to:

- Avoiding unnecessary clones and allocations in decoded image/video paths.
- Keeping `default`, `url`, `video`, and `url + video` feature behavior coherent.
- Returning meaningful typed errors for library-facing APIs.
- Documenting public APIs, error behavior, panic behavior, and safety contracts.
- Testing observable image, sequence, tensor, and feature-gated behavior rather than implementation details.
