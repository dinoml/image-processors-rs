# Processor Compatibility Maintenance Runbook

**Status:** Maintenance reference  
**Backlog baseline:** Complete at the revisions recorded in
[`processor-compatibility-roadmap.md`](processor-compatibility-roadmap.md)

Use this runbook when upstream processors or local compatibility behavior
changes. It is not a list of pending work.

## Ground Rules

- Follow the repository-wide `AGENTS.md` instructions.
- Preserve unrelated changes in a dirty worktree.
- Keep the project Rust-first; Python is allowed only for optional fixture
  generation.
- Do not mirror upstream class hierarchies or introduce hidden downloads.
- Do not retain `FixtureParity` when complete source-linked evidence is absent.
- Treat canonical and `*Pil` backends as distinct numeric contracts whenever
  upstream behavior differs.

## 1. Establish the Change Boundary

Record the old and new source revisions, then identify why the audit is being
run:

- upstream class or alias inventory changed,
- processor defaults or source behavior changed,
- local transforms, layouts, recipes, wrappers, or postprocessing changed,
- fixture schema or tolerance policy changed,
- an optional feature combination changed.

Read the current contracts before editing:

```powershell
Get-Content -Raw AGENTS.md
Get-Content -Raw docs\PARITY.md
Get-Content -Raw docs\ROADMAP.md
Get-Content -Raw docs\backlog\processor-compatibility-roadmap.md
Get-Content -Raw image-processors\src\catalog.rs
Get-Content -Raw image-processors\src\catalog_data.rs
git status --short
```

## 2. Audit Exact Upstream Source

Use clean temporary checkouts outside the repository. Record full 40-character
Git commits and run generators with source imports resolving to those
checkouts. Do not infer numeric provenance from an installed package version.

For Transformers, audit canonical and `*Pil` classes separately. A canonical
Torchvision backend and a Pillow alias may produce different pixels even when
they share a catalog entry.

For Diffusers, include shared processors and the reviewed pipeline-local image
or video processors. Attention processors are not image-processor identities.

## 3. Reconcile the Catalog

Update the private static catalog data only after the source inventory is
known. Verify:

- every canonical class appears once,
- every real alias resolves to the intended canonical entry,
- fabricated aliases are rejected,
- recipe ids and family kinds resolve,
- audit commits are exact,
- compatibility statuses match fixture strength.

The catalog is intentionally static Rust data, not a public serialization or
runtime registry API.

## 4. Generate Evidence

Use the scripts under `scripts/parity/`; command examples and prerequisites are
in [`../PARITY.md`](../PARITY.md). Preserve deterministic inputs and compact
fixtures that still exercise meaningful resize, crop, padding, tiling, frame,
or postprocess behavior.

Every fixture must carry:

- a schema version and generator identity,
- exact upstream source provenance,
- class/backend identity,
- model id or deterministic configuration source,
- input description and effective processor config,
- complete outputs and metadata,
- an explicit comparison policy.

Full floating payloads need recorded absolute and relative tolerances. Shapes,
layouts, dtypes, integer values, bytes, booleans, and metadata are exact unless
the fixture explicitly defines and justifies a different rule.

The evidence tests cap absolute tolerance at `0.06` and relative tolerance at
zero. Any relaxation requires an intentional code-review change as well as a
fixture update.

## 5. Prove Catalog-to-Fixture Coverage

Run both payload parity tests and the suite-level evidence test. The evidence
check must remain bidirectional:

- every `FixtureParity` catalog canonical class and alias is present in a
  checked-in fixture at the catalog audit commit,
- every catalog-bound fixture class resolves to the catalog,
- any auxiliary processor fixture has an explicit family mapping and matching
  audit commit,
- dedicated fixture declarations name real files and matching source commits.

A test that only asserts the `CompatibilityStatus` enum value is not parity
evidence.

## 6. Verify Feature Combinations

Run narrow affected tests first, then the complete matrix:

```powershell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo clippy -p image-processors --all-targets --no-default-features -- -D warnings
cargo test -p image-processors --no-default-features
cargo clippy --workspace --all-targets --features url -- -D warnings
cargo test --workspace --features url
```

With FFmpeg development libraries available:

```powershell
.\scripts\setup-vcpkg-ffmpeg.ps1 -CheckCargo
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Also run `git diff --check` and review the final diff for generated files,
accidental dependency changes, or unrelated edits.

## 7. Update Documentation

After verification:

- update `docs/PARITY.md` with source revisions, fixture schemas, commands, and
  comparison policies,
- update `docs/ROADMAP.md` only for real future product work,
- update the completed-baseline document when the audited boundary changes,
- describe tokenizer/model-schema/training dependencies as explicit scope
  boundaries unless they have intentionally been added to this crate.
