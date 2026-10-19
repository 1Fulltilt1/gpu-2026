#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace {
float *g_d_a = nullptr, *g_d_b = nullptr, *g_d_c = nullptr;
std::size_t g_cap = 0;

void ensureBuffers(std::size_t bytes) {
    if (bytes <= g_cap) return;
    if (g_d_a) cudaFree(g_d_a);
    if (g_d_b) cudaFree(g_d_b);
    if (g_d_c) cudaFree(g_d_c);
    cudaMalloc(&g_d_a, bytes);
    cudaMalloc(&g_d_b, bytes);
    cudaMalloc(&g_d_c, bytes);
    g_cap = bytes;
}

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int col  = blockIdx.x * 32 + threadIdx.x;
    const int row0 = blockIdx.y * 32 + threadIdx.y;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row0 + 8 * r;
        if (row >= n || col >= n) continue;
        float sum = 0.0f;
        const float* arow = a + (std::size_t)row * n;
        #pragma unroll 4
        for (int k = 0; k < n; ++k)
            sum += arow[k] * b[(std::size_t)k * n + col];
        c[(std::size_t)row * n + col] = sum;
    }
}
} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::size_t total = (std::size_t)n * n;
    std::vector<float> c(total, 0.0f);
    if (total == 0) return c;
    std::size_t bytes = total * sizeof(float);

    ensureBuffers(bytes);
    cudaMemcpy(g_d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(32, 8);
    dim3 blocks(n / 32, n / 32);
    NaiveGemmKernel<<<blocks, threads>>>(g_d_a, g_d_b, g_d_c, n);

    cudaMemcpy(c.data(), g_d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}