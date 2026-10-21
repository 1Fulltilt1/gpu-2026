#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 32

__global__ void block_gemm_kernel(const float* __restrict__ d_a,
                                  const float* __restrict__ d_b,
                                  float* __restrict__ d_c,
                                  int n) {
    // 1. Выделение разделяемой памяти под блок A и блок B
    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    int blockCount = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 2. Итерация по всем подблокам (блокам столбца A и строки B)
    for (int k = 0; k < blockCount; ++k) {

        int a_col = k * BLOCK_SIZE + tx;
        if (row < n && a_col < n) {
            s_a[ty][tx] = d_a[row * n + a_col];
        } else {
            s_a[ty][tx] = 0.0f;
        }

        int b_row = k * BLOCK_SIZE + ty;
        if (b_row < n && col < n) {
            s_b[ty][tx] = d_b[b_row * n + col];
        } else {
            s_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum = fmaf(s_a[ty][i], s_b[i][tx], sum);
        }

        __syncthreads();
    }

    // 3. Сохранение итогового результата блока C в глобальную память
    if (row < n && col < n) {
        d_c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = n * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size, 0.0f);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaHostRegister((void*)a.data(), bytes, cudaHostRegisterDefault);
    cudaHostRegister((void*)b.data(), bytes, cudaHostRegisterDefault);
    cudaHostRegister((void*)c.data(), bytes, cudaHostRegisterDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    // 1 поток = 1 элемент матрицы C
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaHostUnregister((void*)a.data());
    cudaHostUnregister((void*)b.data());
    cudaHostUnregister((void*)c.data());

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}