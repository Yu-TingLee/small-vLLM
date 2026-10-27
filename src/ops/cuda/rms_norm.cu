#include <ATen/ceil_div.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "ops/rms_norm_kernels.h"

namespace svllm::ops::cuda_impl {
namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;

// 8 bf16 values make one 16-byte load or store
constexpr int kVecSize = 8;

template <int N>
struct alignas(sizeof(__nv_bfloat16) * N) BF16Vec {
  __nv_bfloat16 v[N];
};

// Definition: Sum of one float per thread across the whole block.
// Formula:    total = sum over threadIdx.x of value, by warp shuffles and then one warp.
// Domain:     blockDim.x a multiple of 32 and at most 1024; called once per kernel.
// Returns:    The same total to every thread.
__device__ float block_sum(float value) {
  __shared__ float warp_sums[kMaxThreads / kWarpSize];
  __shared__ float total;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < blockDim.x / kWarpSize ? warp_sums[lane] : 0.0f;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      total = value;
    }
  }
  __syncthreads();
  return total;
}

// Definition: RMSNorm of one row per block, each thread striding N elements at a time.
// Formula:    out = bf16(x / sqrt(mean(x^2) + eps)) * weight, with the sum accumulated in fp32.
// Domain:     Rows of hidden bf16 values; with N > 1, hidden % N == 0 and 16-byte aligned rows.
// Returns:    Nothing; writes row blockIdx.x of out.
template <int N>
__global__ void rms_norm_kernel(__nv_bfloat16* __restrict__ out,
                                const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ weight, int hidden, float eps) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * hidden;
  const auto* x_row = reinterpret_cast<const BF16Vec<N>*>(x + offset);
  const auto* weight_vecs = reinterpret_cast<const BF16Vec<N>*>(weight);
  auto* out_row = reinterpret_cast<BF16Vec<N>*>(out + offset);
  const int num_vecs = hidden / N;

  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    const BF16Vec<N> xv = x_row[i];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float v = __bfloat162float(xv.v[j]);
      sum_sq += v * v;
    }
  }
  const float inv_rms = rsqrtf(block_sum(sum_sq) / hidden + eps);

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    const BF16Vec<N> xv = x_row[i];
    const BF16Vec<N> wv = weight_vecs[i];
    BF16Vec<N> ov;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(xv.v[j]) * inv_rms);
      ov.v[j] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(wv.v[j]));
    }
    out_row[i] = ov;
  }
}

bool is_vec_aligned(const void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % sizeof(BF16Vec<kVecSize>) == 0;
}

}  // namespace

void rms_norm(const at::Tensor& out, const at::Tensor& x, const at::Tensor& weight, double eps) {
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  const int64_t hidden64 = x.size(-1);
  const int64_t num_rows = x.numel() / hidden64;
  TORCH_CHECK(hidden64 <= kIntMax && num_rows <= kIntMax, "rms_norm: ", num_rows, " rows of ",
              hidden64, " exceed the CUDA kernel's int32 grid and index range");
  const int hidden = static_cast<int>(hidden64);
  if (num_rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard device_guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  const auto* w_ptr = reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr());

  // A row can only be loaded in 16-byte vectors if every row starts on a 16-byte boundary
  const bool vectorized = hidden % kVecSize == 0 && is_vec_aligned(out_ptr) &&
                          is_vec_aligned(x_ptr) && is_vec_aligned(w_ptr);
  const int num_vecs_per_row = vectorized ? hidden / kVecSize : hidden;
  const int num_threads = std::min(kMaxThreads, at::round_up(num_vecs_per_row, kWarpSize));

  const auto kernel = vectorized ? rms_norm_kernel<kVecSize> : rms_norm_kernel<1>;
  kernel<<<static_cast<unsigned int>(num_rows), num_threads, 0, stream>>>(
      out_ptr, x_ptr, w_ptr, hidden, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace svllm::ops::cuda_impl
