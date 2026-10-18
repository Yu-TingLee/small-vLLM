# small-vLLM

A C++/CUDA LLM inference engine on LibTorch: paged KV cache with prefix caching, continuous batching with preemption, our own FlashAttention-2-style prefill and split-KV decode kernels, and an OpenMP CPU backend. Serves Qwen3-0.6B, plus Llama 3.2 1B for comparison. Final project for Parallel Programming.

**Status:** skeleton. The plan is in [docs/design-spec.md](docs/design-spec.md).

## Requirements

- Linux x86_64 with an NVIDIA GPU (sm_86 or sm_89 by default; override with `-DCMAKE_CUDA_ARCHITECTURES=...`), a CUDA toolkit 12.6 or newer (torch 2.14 requires it; `nvcc` on `PATH`, or `CUDACXX` set) and a driver 525 or newer.
- GCC 11 or newer; set `CXX` if your default `g++` is older. CMake checks the GCC and CUDA versions when it configures.
- [uv](https://docs.astral.sh/uv/). It installs Python, torch, CMake and Ninja into `.venv`.

## Quickstart

```bash
uv sync --extra cu126                  # on a machine without a GPU: uv sync --extra cpu
uv run pre-commit install

uv run cmake --preset release          # presets: release, debug, relwithdebinfo (Nsight), cpu
uv run cmake --build --preset release
uv run ctest --preset release          # C++ tests
uv run pytest                          # Python tests; for other presets set SVLLM_BUILD_DIR=build/<preset>

build/release/bin/svllm-bench --device cuda
scripts/download_models.sh             # Qwen3-0.6B; add --llama for Llama 3.2 1B (gated on Hugging Face)
```

## Docs

- [docs/design-spec.md](docs/design-spec.md): what we build and how we evaluate it
- [docs/adr/](docs/adr/): decisions and why
- [AGENTS.md](AGENTS.md): workflow, coding standards and tests, for people and AI agents

## License

Apache-2.0. Design ideas and some code come from [tiny-vllm](https://github.com/jmaczan/tiny-vllm); see [NOTICE](NOTICE).
