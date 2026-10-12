#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int TILE = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * TILE + tx;
    const int row = blockIdx.y * TILE + ty;

    float sum = 0.0f;
    const int num_tiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;

        As[ty][tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = count * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    std::vector<float> result(count);
    cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}
