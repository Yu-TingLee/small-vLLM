# small-vLLM

C++/CUDA LLM inference engine on LibTorch. What we build and why: `docs/design-spec.md`. Scope, by tier: `docs/ROADMAP.md`. This file is the workflow and coding standards for people and agents alike; follow it.

## Commands

```bash
uv sync --extra cu126                  # GPU machine; `--extra cpu` without a GPU
uv run pre-commit install              # once: format and lint on every commit
uv run cmake --preset release          # presets: release, debug, relwithdebinfo, cpu (no CUDA)
uv run cmake --build --preset release
uv run ctest --preset release
SVLLM_BUILD_DIR=build/release uv run pytest
uv run pre-commit run --all-files
```

Without an NVIDIA GPU (e.g. macOS), only the `cpu` preset builds; GPU tests skip themselves.

## Layout

```
src/ops/         op declarations + device dispatch; kernels in cuda/ and cpu/
src/model/       config, safetensors loader, model forward (Qwen3; Llama 3.2 via config flags)
src/engine/      Device, block manager, prefix cache, scheduler, KV offload, engine loop
src/tokenizer/   tokenizers-cpp wrapper
src/apps/        svllm-bench and svllm-serve entry points
bindings/        pybind11 module svllm_ops, for tests and benchmarks only
tests/cpp/       googletest: engine and model logic; CUDA cases skip without a GPU
tests/python/    pytest: ops vs PyTorch, end to end vs Hugging Face
bench/           kernel and throughput benchmarks, plotting
scripts/         model download, environment checks
cmake/           torch discovery, OpenMP, pinned third-party dependencies
```

## Workflow

- Branch off `main` as `<github-handle>/<topic>`. Every change lands through a PR, squash-merged, so the PR title becomes the commit: Conventional Commits without a scope, e.g. `feat: preemption by recompute`.
- CI has no GPU. For PRs touching kernels, the model or the engine, run the GPU tests on a GPU machine and paste the output into the PR with that of `scripts/check_env.sh` and the build preset.
- Record a decision that is hard to reverse in `docs/design-spec.md`, with the reason.

## Coding standards

- C++20, everything in namespace `svllm`. Files `snake_case`; types `PascalCase`; functions and variables `snake_case`; private members end in `_`; constants and enumerators `kPascalCase`.
- `.cu` for kernels, `.cuh` only for device code shared between `.cu` files, `.h`/`.cpp` for everything else.
- Check inputs with `TORCH_CHECK`; host code may throw.
- An op is declared in `src/ops/<op>.h`, with kernels in `src/ops/cuda/<op>.cu` and `src/ops/cpu/<op>.cpp`, and dispatches on its input's device. An op ships both kernels in one PR and errors on an unsupported device. Attention is the exception: its CPU kernels come with the CPU backend, and until then it errors on CPU.
- Read model sizes from `config.json`. Shapes, paths and GPU architectures come from config or arguments.
- Write a comment only when the code can't say it on its own.
- C++ deps: FetchContent in `cmake/dependencies.cmake`, pinned to a release tarball and its SHA256. Python deps: `uv add`, and commit `uv.lock`.

## Tests

Test where it pays off:

- **Each op:** one parametrized pytest against PyTorch at bf16 tolerance over a few shapes, including GQA ratios and block-size edges.
- **The engine:** one end-to-end test that greedy output matches Hugging Face on a few prompts.
- **Block manager and scheduler:** googletest cases for the tricky paths (prefix hits, eviction, admission when blocks run short).
- Getters, glue code and CLI flags need no tests.

## Rules

- Build only what `docs/ROADMAP.md` puts in scope. For a feature it lists as out of scope, or doesn't list, stop and say so; moving it into scope is a PR that edits the roadmap. Bug fixes, refactors, and doc or tooling changes with their own issue need no row.
- You must be able to explain every line you submit. Don't add code the task doesn't need.
- Change only what the task needs. If you change an interface another module calls (an `ops::` signature, the `Model` or `Engine` API), list it under "Interface changes" in the PR description.
- If your change contradicts `docs/design-spec.md`, say so in the PR instead of silently overriding it.
- Never commit model weights or files over 1 MB.
- Never add AI attribution to commits or PRs: no `Co-Authored-By` trailer for an AI agent, no "Generated with" line.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for Yu-TingLee/small-vLLM, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Decisions live in `docs/design-spec.md`, scope in `docs/ROADMAP.md`; no ADRs, no glossary. See `docs/agents/domain.md`.
