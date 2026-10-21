#pragma once

// Per-device kernels behind ops::rms_norm, which validates the inputs and allocates out; include
// this only from the dispatch and the kernels themselves.

#include <ATen/core/Tensor.h>

namespace svllm::ops {

// Definition: The OpenMP kernel of ops::rms_norm, writing into out.
// Formula:    out = bf16(x / sqrt(mean(x^2) + eps)) * weight, with the sum accumulated in fp32.
// Domain:     Inputs already validated by ops::rms_norm; out is contiguous bf16 shaped like x.
// Returns:    Nothing; out holds the normalized rows.
namespace cpu_impl {
void rms_norm(const at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps);
}

// Definition: The CUDA kernel of ops::rms_norm, writing into out on the current stream.
// Formula:    out = bf16(x / sqrt(mean(x^2) + eps)) * weight, with the sum accumulated in fp32.
// Domain:     Inputs already validated by ops::rms_norm; at most 2^31 - 1 rows.
// Returns:    Nothing; out holds the normalized rows once the stream reaches the kernel.
namespace cuda_impl {
void rms_norm(const at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps);
}

}  // namespace svllm::ops
