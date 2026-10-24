#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <mutex>

const int BM = 64;
const int BN = 64;
const int BK = 32;
const int TM = 2;
const int TN = 2;

// Статический кэш памяти
static float* d_a_cache = nullptr;
static float* d_b_cache = nullptr;
static float* d_c_cache = nullptr;
static size_t cached_bytes = 0;
static cudaStream_t stream = nullptr;
static bool initialized = false;
static std::mutex mtx;

__global__ void block_gemm_tiled_shared_kernel(const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                float* __restrict__ c,
                                                int n) {
    // Shared memory для блоков A и B с паддингом против банк-конфликтов
    __shared__ float s_a[BM][BK + 1];
    __shared__ float s_b[BK][BN + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx;

    // Сетка потоков 32x32 покрывает тайл C размером 64x64 (каждый поток считает TM x TN элементов)
    int row_base = by * BM + ty * TM;
    int col_base = bx * BN + tx * TN;

    float accum[TM][TN] = {0.0f};

    int num_tiles = n / BK;

    for (int t = 0; t < num_tiles; ++t) {

        for (int i = thread_id; i < (BM * BK); i += blockDim.x * blockDim.y) {
            int r = i / BK;
            int c_idx = i % BK;
            s_a[r][c_idx] = a[(by * BM + r) * n + (t * BK + c_idx)];
        }

        for (int i = thread_id; i < (BK * BN); i += blockDim.x * blockDim.y) {
            int r = i / BN;
            int c_idx = i % BN;
            s_b[r][c_idx] = b[(t * BK + r) * n + (bx * BN + c_idx)];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += s_a[ty * TM + i][k] * s_b[k][tx * TN + j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = row_base + i;
            int c_idx = col_base + j;
            if (r < n && c_idx < n) {
                c[r * n + c_idx] = accum[i][j];
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = n * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size, 0.0f);
    if (n == 0) return c;

    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }

    if (bytes > cached_bytes) {
        if (d_a_cache) cudaFree(d_a_cache);
        if (d_b_cache) cudaFree(d_b_cache);
        if (d_c_cache) cudaFree(d_c_cache);

        cudaMalloc(&d_a_cache, bytes);
        cudaMalloc(&d_b_cache, bytes);
        cudaMalloc(&d_c_cache, bytes);
        cached_bytes = bytes;
    }

    cudaMemcpyAsync(d_a_cache, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_cache, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(n / BN, n / BM);

    block_gemm_tiled_shared_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a_cache, d_b_cache, d_c_cache, n);

    cudaMemcpyAsync(c.data(), d_c_cache, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return c;
}