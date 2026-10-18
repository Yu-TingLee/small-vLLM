# One model, ops dispatched by tensor device

Like PyTorch, there is a single model implementation and the engine is built for one `Device` (CUDA by default, or CPU); every op under `ops::` picks its CUDA or OpenMP kernel from the input tensor's device. We rejected separate CUDA and CPU model classes because every model fix (QK-norm, Llama RoPE scaling) would have to land twice, and the CPU path would drift from the GPU one. The scheduler, block manager and KV pool are device-agnostic.

## Consequences

- Adding an op means writing a declaration plus one kernel per device; a missing CPU kernel is a build or runtime error, not a silent fallback.
- GEMM uses `torch::matmul` on both devices (cuBLAS on GPU, MKL/oneDNN on CPU). Hand-written GEMMs are deferred until a benchmark shows we need them.
