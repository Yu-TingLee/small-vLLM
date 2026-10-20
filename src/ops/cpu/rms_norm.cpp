#include "ops/rms_norm.h"

#include <c10/util/BFloat16.h>

#include <cmath>
#include <cstdint>

namespace svllm::ops::cpu {

void rms_norm(const at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps) {
  const int64_t hidden = x.size(-1);
  const int64_t rows = x.numel() / hidden;
  const auto* x_data = x.const_data_ptr<at::BFloat16>();
  const auto* w = weight.const_data_ptr<at::BFloat16>();
  auto* out_data = out.mutable_data_ptr<at::BFloat16>();
  const float eps_f = static_cast<float>(eps);

#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    const at::BFloat16* x_row = x_data + row * hidden;
    at::BFloat16* out_row = out_data + row * hidden;

    // The reduction clause lets the compiler reorder the fp32 sum into SIMD lanes
    float sum_sq = 0.0f;
#pragma omp simd reduction(+ : sum_sq)
    for (int64_t i = 0; i < hidden; ++i) {
      const float v = x_row[i];
      sum_sq += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) + eps_f);

    for (int64_t i = 0; i < hidden; ++i) {
      // Round to bf16 before scaling by the weight, as Hugging Face's Qwen3 and Llama do
      const at::BFloat16 normed = static_cast<float>(x_row[i]) * inv_rms;
      out_row[i] = static_cast<float>(normed) * static_cast<float>(w[i]);
    }
  }
}

}  // namespace svllm::ops::cpu
