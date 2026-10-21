# small-vLLM

A C++/CUDA LLM inference engine on LibTorch: paged KV cache with prefix caching, continuous batching with preemption, our own FlashAttention-2-style prefill and split-KV decode kernels, and an OpenMP CPU backend. Serves Qwen3-0.6B, plus Llama 3.2 1B for comparison. The design is in [docs/design-spec.md](docs/design-spec.md).

## Requirements

- Linux with an NVIDIA GPU, CUDA 12.6+ and GCC 11+
- [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
uv sync --extra cu126                               # on a machine without a GPU: uv sync --extra cpu
uv run pre-commit install

uv run python scripts/download_tokenizers.py        # Download required test tokenizers for C++ tokenizer tests

uv run cmake --preset release                       # presets: release, debug, relwithdebinfo (Nsight), cpu
uv run cmake --build --preset release
uv run ctest --preset release                       # C++ tests
uv run pytest                                       # Python tests; for other presets set SVLLM_BUILD_DIR=build/<preset>

build/release/bin/svllm-bench --device cuda
scripts/download_models.sh                          # Qwen3-0.6B; add --llama for Llama 3.2 1B (gated on Hugging Face)
```

## License

Apache-2.0; see [NOTICE](NOTICE).
