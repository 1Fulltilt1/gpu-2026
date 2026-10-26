#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 256;

// Running (max, sum) pair for the online softmax algorithm. The sum is
// accumulated relative to the current maximum, so a single pass over a row is
// enough to get both the maximum and the sum of exponents.
struct MaxSum {
    float max;
    float sum;
};

__device__ __forceinline__ MaxSum MergeMaxSum(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.max, b.max);
    if (m == -FLT_MAX) return {m, 0.0f};
    return {m, a.sum * __expf(a.max - m) + b.sum * __expf(b.max - m)};
}

__device__ __forceinline__ void AddElement(MaxSum& v, float x) {
    if (x > v.max) {
        v.sum = v.sum * __expf(v.max - x) + 1.0f;
        v.max = x;
    } else {
        v.sum += __expf(x - v.max);
    }
}

__device__ __forceinline__ MaxSum WarpReduce(MaxSum v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        MaxSum other;
        other.max = __shfl_down_sync(0xffffffffu, v.max, offset);
        other.sum = __shfl_down_sync(0xffffffffu, v.sum, offset);
        v = MergeMaxSum(v, other);
    }
    return v;
}

// Block-wide reduction: first inside every warp with shuffles, then the first
// warp combines one partial result from each warp.
__device__ __forceinline__ MaxSum BlockReduce(MaxSum v, MaxSum* warp_results) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    v = WarpReduce(v);
    if (lane == 0) warp_results[warp] = v;
    __syncthreads();

    const int warp_count = blockDim.x / kWarpSize;
    MaxSum block = (threadIdx.x < warp_count) ? warp_results[threadIdx.x] : MaxSum{-FLT_MAX, 0.0f};
    if (warp == 0) block = WarpReduce(block);

    __shared__ MaxSum result;
    if (threadIdx.x == 0) result = block;
    __syncthreads();
    return result;
}

// One CUDA block processes one row.
__global__ void SoftmaxKernel(const float* __restrict__ input, float* __restrict__ output, int row_size) {
    __shared__ MaxSum warp_results[kMaxThreads / kWarpSize];

    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * row_size;
    const float* row_in = input + base;
    float* row_out = output + base;

    MaxSum local{-FLT_MAX, 0.0f};
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        AddElement(local, row_in[i]);
    }

    const MaxSum row = BlockReduce(local, warp_results);
    const float inv_sum = (row.sum > 0.0f) ? 1.0f / row.sum : 0.0f;

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_out[i] = __expf(row_in[i] - row.max) * inv_sum;
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const std::size_t count = input.size();
    if (count == 0 || row_count <= 0) return std::vector<float>(count);

    const std::size_t row_size = count / static_cast<std::size_t>(row_count);
    if (row_size == 0) return std::vector<float>(count);

    const std::size_t bytes = count * sizeof(float);

    float* dev_input = nullptr;
    float* dev_output = nullptr;
    cudaMalloc(&dev_input, bytes);
    cudaMalloc(&dev_output, bytes);

    cudaMemcpy(dev_input, input.data(), bytes, cudaMemcpyHostToDevice);

    int block_size = kWarpSize;
    while (block_size < kMaxThreads && static_cast<std::size_t>(block_size) < row_size) {
        block_size <<= 1;
    }

    SoftmaxKernel<<<row_count, block_size>>>(dev_input, dev_output, static_cast<int>(row_size));

    std::vector<float> result(count);
    cudaMemcpy(result.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_output);
    return result;
}
