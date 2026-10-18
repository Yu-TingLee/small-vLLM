# small-vLLM

C++/CUDA LLM inference engine on LibTorch. What we build: `docs/design-spec.md`. How we work and the coding standards: `CONTRIBUTING.md`; follow it.

## Commands

```bash
uv sync --extra cu126                  # GPU machine; `--extra cpu` without a GPU
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
