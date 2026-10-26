#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <vector>

namespace {

constexpr int kMaxThreads = 256;

// One CUDA block processes one row.
__global__ void SoftmaxKernel(const float* __restrict__ input, float* __restrict__ output, int row_size) {
    __shared__ float shared[kMaxThreads];

    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * row_size;
    const float* row_in = input + base;
    float* row_out = output + base;

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        local_sum += __expf(row_in[i] - row_max);
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / shared[0];

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_out[i] = __expf(row_in[i] - row_max) * inv_sum;
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

    int block_size = 32;
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
