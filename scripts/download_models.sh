#!/usr/bin/env bash
# Downloads model weights into the Hugging Face cache and prints their local paths.
# Usage: scripts/download_models.sh [--llama]
set -euo pipefail

QWEN3_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
# The revision tiny-vllm benchmarks against.
LLAMA_REVISION=898999bd25b40516fce5a5b8f0948f4c81c650bc

uv run hf download Qwen/Qwen3-0.6B --revision "$QWEN3_REVISION" --include "*.json" "*.safetensors" "*.txt"

if [[ "${1:-}" == "--llama" ]]; then
  # Gated: accept the license on huggingface.co, then run `uv run hf auth login` once.
  uv run hf download meta-llama/Llama-3.2-1B-Instruct --revision "$LLAMA_REVISION" \
    --include "*.json" "*.safetensors"
fi
