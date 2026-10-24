#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>

// Каждый блок 16x16 потоков вычисляет тайл размером 32x32 в матрице C.
// Каждый поток вычисляет подсетку 2x2 элементов.
#define TILE_DIM 32
#define THREAD_DIM 16
#define SHMEM_PAD 1

__global__ void ultra_gemm_kernel(const float* __restrict__ d_a,
                                  const float* __restrict__ d_b,
                                  float* __restrict__ d_c,
                                  int n) {
    __shared__ float s_a[TILE_DIM][TILE_DIM + SHMEM_PAD];
    __shared__ float s_b[TILE_DIM][TILE_DIM + SHMEM_PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float c_sum[2][2] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

    int numTiles = (n + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int a_row = row + i * THREAD_DIM;
                int a_col = t * TILE_DIM + tx + j * THREAD_DIM;
                
                int b_row = t * TILE_DIM + ty + i * THREAD_DIM;
                int b_col = col + j * THREAD_DIM;

                s_a[ty + i * THREAD_DIM][tx + j * THREAD_DIM] = 
                    (a_row < n && a_col < n) ? d_a[a_row * n + a_col] : 0.0f;

                s_b[ty + i * THREAD_DIM][tx + j * THREAD_DIM] = 
                    (b_row < n && b_col < n) ? d_b[b_row * n + b_col] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {

            float a0 = s_a[ty][k];
            float a1 = s_a[ty + THREAD_DIM][k];

            float b0 = s_b[k][tx];
            float b1 = s_b[k][tx + THREAD_DIM];

            c_sum[0][0] = fmaf(a0, b0, c_sum[0][0]);
            c_sum[0][1] = fmaf(a0, b1, c_sum[0][1]);
            c_sum[1][0] = fmaf(a1, b0, c_sum[1][0]);
            c_sum[1][1] = fmaf(a1, b1, c_sum[1][1]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int out_row = row + i * THREAD_DIM;
            int out_col = col + j * THREAD_DIM;
            if (out_row < n && out_col < n) {
                d_c[out_row * n + out_col] = c_sum[i][j];
            }
        }
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    // Конфигурация сетки: блоки 16x16, но каждый обрабатывает 32x32 элементы
    dim3 threadsPerBlock(THREAD_DIM, THREAD_DIM);
    dim3 numBlocks((n + TILE_DIM - 1) / TILE_DIM,
                   (n + TILE_DIM - 1) / TILE_DIM);

    ultra_gemm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, n);

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