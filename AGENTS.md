# small-vLLM

C++/CUDA LLM inference engine on LibTorch. What we build: `docs/design-spec.md`. This file is the workflow and coding standards for people and agents alike; follow it.

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
tests/cpp/       googletest: engine logic, CPU only
tests/python/    pytest: ops vs PyTorch, end to end vs Hugging Face
bench/           kernel and throughput benchmarks, plotting
scripts/         model download, environment checks
cmake/           torch discovery, OpenMP, pinned third-party dependencies
docs/adr/        decisions; read the ones touching your area before changing it
```

## Workflow

- Branch off `main` as `<github-handle>/<topic>`. Every change lands through a PR, squash-merged, so the PR title becomes the commit: Conventional Commits with the area as scope, e.g. `feat(scheduler): preemption by recompute`.
- CI has no GPU. For PRs touching kernels, the model or the engine, run the GPU tests on a GPU machine and paste the output, with the GPU name, into the PR. Benchmark numbers for the report come from the RTX 3090.
- Record a decision that is hard to reverse as an ADR in `docs/adr/`.

## Coding standards

- C++20, everything in namespace `svllm`. Files `snake_case`; types `PascalCase`; functions and variables `snake_case`; private members end in `_`; constants and enumerators `kPascalCase`.
- `.cu` for kernels, `.cuh` only for device code shared between `.cu` files, `.h`/`.cpp` for everything else.
- Check inputs with `TORCH_CHECK`; host code may throw.
- An op is declared in `src/ops/<op>.h`, with kernels in `src/ops/cuda/<op>.cu` and `src/ops/cpu/<op>.cpp`, and dispatches on its input's device (ADR-0002). Every op has both kernels and errors on an unsupported device.
- Read model sizes from `config.json`. Shapes, paths and GPU architectures come from config or arguments.
- Write a comment only when the code can't say it on its own.
- C++ deps: FetchContent in `cmake/dependencies.cmake`, pinned to a release tarball and its SHA256. Python deps: `uv add`, and commit `uv.lock`.

## Tests

Test where it pays off:

- **Each op:** one parametrized pytest against PyTorch at bf16 tolerance over a few shapes, including GQA ratios and block-size edges.
- **The engine:** one end-to-end test that greedy output matches Hugging Face on a few prompts.
- **Block manager and scheduler:** googletest cases for the tricky paths (prefix hits, eviction, preemption).
- Getters, glue code and CLI flags need no tests.

## Rules

- You must be able to explain every line you submit. Don't add code the task doesn't need.
- Change only what the task needs. If you change an interface another module calls (an `ops::` signature, the `Model` or `Engine` API), list it under "Interface changes" in the PR description.
- If your change contradicts an ADR, say so in the PR instead of silently overriding it.
- Never commit model weights or files over 1 MB.
- Never add AI attribution to commits or PRs: no `Co-Authored-By` trailer for an AI agent, no "Generated with" line.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for Yu-TingLee/small-vLLM, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

ADRs only, in `docs/adr/`; no glossary. See `docs/agents/domain.md`.
