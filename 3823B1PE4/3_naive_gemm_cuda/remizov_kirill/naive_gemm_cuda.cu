#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kThreads = 16;

// Naive GEMM: one thread computes a single element of C.
// C[row][col] = sum_k A[row][k] * B[k][col]
__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }
    const std::size_t count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = count * sizeof(float);

    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice);

    const int blocks = (n + kThreads - 1) / kThreads;
    NaiveGemmKernel<<<dim3(blocks, blocks), dim3(kThreads, kThreads)>>>(da, db, dc, n);

    std::vector<float> result(count);
    cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return result;
}
