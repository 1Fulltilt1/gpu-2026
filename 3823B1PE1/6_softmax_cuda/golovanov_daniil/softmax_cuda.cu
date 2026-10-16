#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 128;

__global__ void SoftmaxKernel(const float* input, float* output,
                              int row_count, int row_size) {
    const int row = blockIdx.x;
    if (row >= row_count) return;

    const float* row_in  = input  + static_cast<std::size_t>(row) * row_size;
    float*       row_out = output + static_cast<std::size_t>(row) * row_size;

    extern __shared__ float sdata[];

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < row_size; j += blockDim.x) {
        local_max = fmaxf(local_max, row_in[j]);
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < row_size; j += blockDim.x) {
        const float e = __expf(row_in[j] - row_max);
        sdata[threadIdx.x + blockDim.x] = e;
        local_sum += e;
    }

    __syncthreads();
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / sdata[0];

    for (int j = threadIdx.x; j < row_size; j += blockDim.x) {
        row_out[j] = __expf(row_in[j] - row_max) * inv_sum;
    }
}

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument("invalid row_count");
    }
    const int row_size = static_cast<int>(input.size() / row_count);
    if (row_size <= 0) {
        throw std::invalid_argument("row_size must be positive");
    }

    std::vector<float> output(input.size(), 0.0f);
    if (input.empty()) return output;

    float *d_in = nullptr, *d_out = nullptr;
    const std::size_t bytes = input.size() * sizeof(float);

    CheckCuda(cudaMalloc(&d_in, bytes), "cudaMalloc(input)");
    try {
        CheckCuda(cudaMalloc(&d_out, bytes), "cudaMalloc(output)");
        CheckCuda(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");

        const int threads = kThreads;
        const int blocks = row_count;
        const std::size_t shmem = static_cast<std::size_t>(threads) * 2 * sizeof(float);

        SoftmaxKernel<<<blocks, threads, shmem>>>(d_in, d_out, row_count, row_size);
        CheckCuda(cudaGetLastError(), "SoftmaxKernel launch");
        CheckCuda(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost), "copy output");
    } catch (...) {
        cudaFree(d_out); cudaFree(d_in);
        throw;
    }
    CheckCuda(cudaFree(d_out), "cudaFree(output)");
    CheckCuda(cudaFree(d_in), "cudaFree(input)");
    return output;
}