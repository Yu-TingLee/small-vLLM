# small-vLLM roadmap

What is in scope, at which tier, and what is out. Find a feature here before building it. If it is out of scope or not listed, stop and say so; moving a feature into scope takes a PR that edits this file. Each feature's issue holds its "done when"; `docs/design-spec.md` holds the architecture.

- **Must:** the engine and every number in the report.
- **Should:** the demo, the CPU backend and the optimizations we plan to measure.
- **Stretch:** only once Must and Should are done.

## Limits

These hold for every feature.

- **Models:** Qwen3-0.6B and Llama 3.2 1B Instruct. Sizes and flags come from `config.json`, EOS ids from `generation_config.json`.
- **Precision:** bf16 weights, activations and KV cache, on both devices. Kernels accumulate in fp32. Ops reject other dtypes with `TORCH_CHECK`. The FP8 Stretch items are the only exceptions.
- **GPUs:** sm_86 (RTX 3090) and sm_89 (RTX 4090).
- **GEMM:** `torch::matmul`.
- **Sizes:** `max_model_len`, `max_num_seqs` and `max_num_batched_tokens` are flags, defaulting to 4096, 256 and 8192; startup requires `max_num_batched_tokens >= max_model_len`. `block_size` is 8, 16 or 32. Tests cover contexts up to 4096 tokens.
- **Decode kernels are graph-capturable:** no allocation, no host sync, and a launch grid fixed by the maximum sizes rather than by the batch, so CUDA graphs need no kernel rework.

## Must

| Feature | Scope | Issues |
| --- | --- | --- |
| Model forward | Qwen3 on LibTorch tensors, weights from safetensors | #2, #4 |
| Llama 3.2 1B | No QK-norm and llama3 RoPE scaling, behind config flags | #21 |
| Tokenizer | tokenizers-cpp | #3 |
| Ops dispatched by device | The pattern, with RMSNorm as the first op | #12 |
| Paged KV cache | Block manager, block tables, paged KV in the forward | #6, #7, #8 |
| Prefix caching | Reuse KV blocks of a shared prompt prefix | #10 |
| Continuous batching | Each step is a batched prefill or a decode, prefill first. A request is admitted only when blocks for its prompt plus `max_tokens` are free, and refused when that exceeds `max_model_len`. | #9 |
| Prefill attention | FA2-style kernel on paged KV | #15 |
| Decode attention | Split-KV paged kernel, GQA-aware | #16 |
| Generation | Greedy; stop on any EOS id or at `max_tokens` | #5 |
| `svllm-bench` | Offline request sets through the engine in-process, JSON results, plots. An `ignore_eos` request option and a static-batching mode, for the baselines. | #5, #22 |

## Should

| Feature | Scope | Issues |
| --- | --- | --- |
| Fused ops | RoPE with QK-norm; SiLU-and-mul | #13, #14 |
| Sampling | Temperature, top-k, top-p and `seed`, on the engine's device | #19 |
| CUDA graphs | Decode steps captured at batch sizes 1, 2, 4, 8, then every 8 up to `max_num_seqs`; batches padded up | #20 |
| `svllm-serve` | Engine on its own thread; `POST /v1/chat/completions` streamed (SSE) and non-streamed; `GET /stats` with tokens/s, KV usage and prefix hit rate; one demo page; localhost, no auth. Chat formats for Qwen3 (thinking off) and Llama 3 written in C++, picked by `model_type`. | #24 |
| CPU backend | OpenMP prefill and decode attention, so the whole model runs on CPU. Every other op ships its CPU kernel with its CUDA kernel. | #17, #18 |
| KV offload to host RAM | Evicted prefix blocks kept in host RAM and copied back on reuse | #23 |

## Stretch

| Feature | Scope | Issues |
| --- | --- | --- |
| Chunked prefill | Long prompts split into chunks and batched with decodes | #25 |
| Data-parallel replicas | Two engine copies on the 2× RTX 4090 server | #26 |
| Preemption by recompute | Replaces reserving `max_tokens` at admission; recompute only | #11 |
| FP8 weights | Through LibTorch ops (`_scaled_mm`); sm_89 only | #42 |
| FP8 KV cache | E4M3, dequantized inside our attention kernels | #43 |

## Out of scope

| Feature | Why |
| --- | --- |
| Tensor and pipeline parallelism, multi-node | Single-GPU engine; data-parallel replicas are the only multi-GPU work |
| Speculative decoding | No smaller draft model exists for either model |
| Preemption by swapping KV to host RAM | Recompute only; KV offload covers evicted prefix blocks |
| fp16 and fp32 inference | Both models ship in bf16 |
| Weight quantization outside LibTorch ops, our own GEMM kernels | GEMM stays in `torch::matmul` |
| LoRA, multimodal input, MoE, other models | Two dense text models |
| Beam search, `n > 1` | One sequence per request |
| Stop strings, logprobs, penalties, grammar-constrained output | Generation stops on EOS or `max_tokens` only |
| Jinja chat templates | Two hand-written chat formats |
| `/v1/completions`, embeddings, tool calls, several models per server, auth | `svllm-serve` exists for the demo |
| Python on the serving path, `torch.compile` | The engine is C++; Python is for tests and benchmarks |
| ROCm, Metal, Windows | NVIDIA GPUs on Linux, plus the CPU-only build |
