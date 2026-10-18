# Contributing

Setup and build commands are in the [README](README.md). Install the git hooks once with `uv run pre-commit install`.

## Workflow

- Branch off `main` as `<github-handle>/<topic>` and keep branches short-lived.
- Every change reaches `main` through a PR. It needs one approval from someone other than the author and green CI (`pre-commit`, `pr-title`, `cpu-tests`). New pushes dismiss earlier approvals.
- PR titles follow [Conventional Commits](https://www.conventionalcommits.org/), with the area as an optional scope: `feat(scheduler): preemption by recompute`, `fix(ops): GQA index in decode`. Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `ci`. PRs are squash-merged, so the title becomes the commit on `main`.
- CI has no GPU. For PRs that touch kernels, the model or the engine, run the GPU tests on a GPU machine and paste the output, with the GPU name, into the PR.
- Benchmark numbers for the report come from the RTX 3090 machine.
- Record decisions that are hard to reverse as an ADR in `docs/adr/`.

## Coding standards

### C++ and CUDA

- C++20, everything in namespace `svllm`.
- Files `snake_case`; types `PascalCase`; functions and variables `snake_case`; private members end in `_`; constants and enumerators `kPascalCase`.
- `.cu` for kernels, `.cuh` only for device code shared between `.cu` files, `.h`/`.cpp` for everything else.
- Check inputs with `TORCH_CHECK`, not `assert`. Host code may throw.
- An op is declared in `src/ops/<op>.h`, with kernels in `src/ops/cuda/<op>.cu` and `src/ops/cpu/<op>.cpp`, and dispatches on its input tensor's device (ADR-0002). Every op has both kernels; never fall back silently to another device.
- Read model sizes from `config.json`. Never hardcode shapes, paths or GPU architectures.
- Comments are sparse: write one only when the code can't say it on its own.

### Python

`ruff` formats and lints; line length 100.

### Formatting

pre-commit runs clang-format and ruff. CI fails on any formatting diff, so style never comes up in review.

## Tests

Test where it pays off, not everything:

- **Each op:** one parametrized pytest against PyTorch at bf16 tolerance over a few shapes, including GQA ratios and block-size edges.
- **The engine:** one end-to-end test that greedy output matches Hugging Face on a few prompts.
- **Block manager and scheduler:** googletest cases for the tricky paths (prefix hits, eviction, preemption).
- **Skip:** getters, glue code, CLI flags.

Tests that need a GPU skip themselves when none is present, so CI runs only the CPU side.

## Dependencies

- **C++:** add to `cmake/dependencies.cmake` with FetchContent, pinned to a release tarball and its SHA256.
- **Python:** `uv add <package>` (or `uv add --dev`), and commit `uv.lock`.
- **Model weights:** `scripts/download_models.sh`. Never commit them.

## Code adapted from tiny-vllm

[tiny-vllm](https://github.com/jmaczan/tiny-vllm) is a reference, not a dependency. If you copy code from it, start the file (or the copied block) with:

```cpp
// Adapted from jmaczan/tiny-vllm (Apache-2.0), modified.
```

The credit in `NOTICE` already covers it.
