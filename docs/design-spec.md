# small-vLLM design spec

This spec says what we build and how we evaluate it. Which features are in scope, at which tier, is in [ROADMAP.md](ROADMAP.md).

## Goal and scope

We build **small-vLLM**: a C++/CUDA serving engine with a paged KV cache, continuous batching, our own attention kernels and an OpenMP CPU backend.

- **Tools:** CUDA, OpenMP, pthreads, on LibTorch (the C++ API of PyTorch) for tensors and GEMM.
- **Models:** Qwen3-0.6B by default; Llama 3.2 1B Instruct for the comparison with tiny-vllm.
- **Output:** a paper-style report. Experiments run through an offline benchmark binary and are plotted from its JSON results.
- **Demo:** a few concurrent chat streams in a browser with live tokens/s, KV usage and cache hits.

## Comparison with tiny-vllm

[tiny-vllm](https://github.com/jmaczan/tiny-vllm) is the closest project: a single-GPU C++/CUDA engine with paged KV and continuous batching. It is a reference only; we start a fresh codebase and credit any code we adapt in `NOTICE`.

| Component | tiny-vllm | small-vLLM |
| --- | --- | --- |
| Model | Llama 3.2 1B; sizes hardcoded | Qwen3-0.6B; Llama 3.2 1B through `config.json` flags; sizes read from `config.json` |
| Tensors | Raw CUDA pointers | LibTorch tensors |
| Tokenizer | Python (Hugging Face) | tokenizers-cpp |
| Block manager | Paged KV cache | Paged KV cache + prefix caching |
| Scheduler | Continuous batching; each new prompt is prefilled alone while the batch waits | Continuous batching; waiting prompts prefilled together in one step; preemption by recompute |
| Prefill attention | Online softmax, FlashAttention-like | FA2-style kernel on paged KV |
| Decode attention | PagedAttention kernel | Split-KV paged kernel; GQA-aware |
| GEMM | cuBLAS | `torch::matmul` (cuBLAS on GPU, MKL/oneDNN on CPU) |
| RMSNorm, RoPE, SiLU | Separate CUDA kernels | Fused CUDA kernels |
| Sampling | Greedy argmax on CPU | On GPU |
| CUDA graphs | No | Yes, around our kernels |

### Only in small-vLLM

| Feature | What it is |
| --- | --- |
| Offline benchmark | `svllm-bench`: runs a request set through the engine in-process and writes JSON |
| Serving | `svllm-serve`: engine loop on its own thread, OpenAI-style chat HTTP, one static demo page |
| CPU backend | The whole model on CPU: an OpenMP kernel for every op; GEMM through `torch::matmul` |
| KV offload to host RAM | Evicted prefix KV is kept in host RAM and copied back on reuse instead of recomputed |
| Chunked prefill | Long prompts split into chunks and batched with decodes |
| Data-parallel replicas | Two engine copies on the 2× RTX 4090 lab server |

## Architecture

C++20 and CUDA throughout, on the LibTorch bundled with the pinned pip `torch`.

- **One model, ops dispatched by device.** The model forward is written once, so a model fix (QK-norm, Llama RoPE scaling) lands once and the CPU path can't drift from the GPU one. Each op under `ops::` (attention, RMSNorm, RoPE, SiLU, sampling) has a CUDA and an OpenMP kernel and picks one from its input tensor's device, like PyTorch; a missing kernel is an error, not a silent fallback. The engine is built for one `c10::Device`, `cuda:0` by default or `cpu`, the type tensors, ops and the safetensors loader already take; weights, the KV pool and activations live there.
- **Engine.** Block manager with prefix caching, scheduler with continuous batching and preemption by recompute, KV offload to host RAM, and the engine loop. All device-agnostic.
- **Model.** Loads safetensors weights and `config.json` directly; Qwen3 by default, with Llama 3.2 differences (no QK-norm, llama3 RoPE scaling) behind config flags.
- **Front ends.** `svllm-bench` (offline, for experiments) and `svllm-serve` (localhost HTTP plus one HTML page, for the demo). Both are thin layers over the same engine API. Changing the device or attention backend restarts the engine with new flags.
- **Python binding (tests and benchmarks only).** The pybind11 module `svllm_ops` exposes our ops and the engine, so tests can compare them with PyTorch and Hugging Face, and benchmarks with flash-attn and vLLM kernels. Python is never on the serving path.

## Hardware

| Machine | GPU | Use |
| --- | --- | --- |
| Benchmark machine | 1× RTX 3090 (sm_86), always available | Every number in the report, except DP replicas |
| Lab server (shared, when free) | 2× RTX 4090 (sm_89) | Development; data-parallel replicas |

We build for sm_86 and sm_89 with a CUDA 12.6+ toolkit, matching the pinned torch wheel (cu126); any driver from 525 runs it. Laptops are for editing and the CPU-only build; GPU work runs over SSH, and the demo page is reached through an SSH port forward.

## Related work

| Project | What it is | How small-vLLM differs |
| --- | --- | --- |
| [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) | About 1,200 lines of Python on PyTorch; paged KV, continuous batching, prefix caching, tensor parallelism; attention via the flash-attn library | Our own attention kernels; C++ engine; OpenMP CPU backend; KV offload to host RAM |
| [tiny-vllm](https://github.com/jmaczan/tiny-vllm) | C++/CUDA engine and course | See Comparison with tiny-vllm |

## Reading list

| Read | Why we need it | In vLLM? |
| --- | --- | --- |
| [tiny-vllm](https://github.com/jmaczan/tiny-vllm) | Closest project; C++/CUDA baseline | No |
| [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) | Reference design; Python baseline | No; a minimal re-implementation of vLLM |
| [PagedAttention](https://arxiv.org/abs/2309.06180) | Block manager, block tables | Yes; the core of vLLM (KV cache manager) |
| [Continuous Batching](https://www.usenix.org/system/files/osdi22-yu.pdf) | Continuous batching in our scheduler | Yes; scheduler |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Our prefill kernel | Yes; default attention backend on NVIDIA GPUs |
| [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | Split-KV in our decode kernel | Yes; split-KV in vLLM's paged attention v2 kernel |
| [FlashInfer](https://arxiv.org/abs/2501.01005) | Paged KV layout for our kernels | Yes; optional attention backend |
| [LMCache](https://github.com/lmcache/lmcache) | KV offload to host RAM | Yes; plugs in as a KV connector |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefill (stretch) | Yes; scheduler, on by default |

## Evaluation plan

Ours against a baseline on the benchmark machine.

| Experiment | Metric | Baseline |
| --- | --- | --- |
| Decode attention | GB/s as % of peak; sweep batch, context, block size, splits, GQA group | flash-attn, vLLM |
| Prefill attention | TFLOP/s vs prompt length | flash-attn |
| Throughput | Tokens/s, nano-vLLM's bench (256 requests, 100–1024 tokens) | nano-vLLM, vLLM |
| Throughput on Llama 3.2 1B | Tokens/s, same bench | tiny-vllm |
| Latency | TTFT, TPOT | Static batching |
| CPU backend | Speedup vs threads | 1 thread; llama.cpp (same precision) |
| KV offload | Hit rate; TTFT on multi-turn chat | GPU-only prefix cache |
| DP replicas (stretch) | Tokens/s, 1 vs 2 replicas, on the lab server | 1 replica |

- **Decode / prefill:** decode is limited by memory bandwidth, so we report GB/s. Prefill is limited by compute, so we report TFLOP/s.
- **GQA group:** how many query heads share one KV head (2 on Qwen3-0.6B, 4 on Llama 3.2 1B).
- **TTFT:** time to first token. **TPOT:** time per output token after that.
- **Static batching:** a batch starts and finishes together, so new requests wait.

## Correctness

- Every op matches PyTorch within bf16 tolerance, on both devices.
- Greedy output matches Hugging Face end to end on a few prompts.
- Block manager and scheduler edge cases (prefix hits, eviction, preemption) have C++ unit tests.
