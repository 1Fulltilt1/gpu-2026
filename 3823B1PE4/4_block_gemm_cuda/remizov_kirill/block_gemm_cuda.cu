#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kTile = 16;

// One CUDA block computes a kTile x kTile block of C, every thread produces a
// single element. For each k-tile a block of A and a block of B are staged in
// shared memory so each element is read from global memory only once per tile.
__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float block_a[kTile][kTile];
    __shared__ float block_b[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    for (int k0 = 0; k0 < n; k0 += kTile) {
        block_a[ty][tx] = (row < n && k0 + tx < n) ? a[row * n + k0 + tx] : 0.0f;
        block_b[ty][tx] = (k0 + ty < n && col < n) ? b[(k0 + ty) * n + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum += block_a[ty][k] * block_b[k][tx];
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

    const int blocks = (n + kTile - 1) / kTile;
    BlockGemmKernel<<<dim3(blocks, blocks), dim3(kTile, kTile)>>>(da, db, dc, n);

    std::vector<float> result(count);
    cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return result;
}
