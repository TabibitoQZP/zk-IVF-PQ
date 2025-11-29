# Repository Guidelines

This repository implements zero-knowledge IVF–PQ primitives as a Rust `cdylib` exposed to Python via `pyo3`/`maturin`. Use this guide when adding or modifying code.

## Project Structure & Module Organization

- Core Rust crate lives in `src/`, with modules such as `ivf_pq`, `ivf_flat`, `merkle_commit`, and `utils`; `lib.rs` exposes the public API.
- Python integration, tests, and helpers live in `tests/`, `bench/`, and `bench_free_bench/` (pipelines, benchmarks, and scripts).
- Shell scripts for experiments and benchmarks are under `scripts/` (for example `scripts/ivf-pq.sh`).
- Large data files and embeddings go in `data/`; avoid committing new large binaries when possible.

## Build, Test, and Development Commands

- Build Rust library only: `cargo build --release`.
- Build Python extension with maturin (local dev): `maturin develop` (installs into current environment).
- Run Python-focused pipeline/tests: use the appropriate entry in `tests/` (for example `python tests/pipeline.py`) or an experiment script in `scripts/`.
- Format Rust code: `cargo fmt`; check for common issues with `cargo clippy`.

## Coding Style & Naming Conventions

- Rust: follow standard Rust 2021 style, 4-space indentation; run `cargo fmt` before pushing.
- Python: prefer Black-style formatting (4-space indentation, snake_case for functions and variables, PascalCase for classes).
- Modules and files should use descriptive, lower_snake_case names (for example `ivf_pq_verify`, `merkle_commit`).

## Testing Guidelines

- Prefer small, focused tests in Python under `tests/` and Rust tests collocated with modules when needed.
- Name test helpers descriptively (for example `*_test.py` or `test_*` functions).
- Before submitting, run at least the relevant Python pipeline (for example `python tests/pipeline.py`) and any affected Rust tests or benches.

## Commit & Pull Request Guidelines

- Keep commits scoped and descriptive (for example `fix ivf_pq verify path` rather than `still have some bugs`).
- Reference related issues in commit messages or PR descriptions when applicable.
- Pull requests should describe the motivation, summarize key changes, and note how you tested (commands and datasets). Include benchmark details for performance-sensitive changes.

