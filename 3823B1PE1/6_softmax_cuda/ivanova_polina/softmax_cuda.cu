#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>

static float* d_in_cache = nullptr;
static float* d_out_cache = nullptr;
static size_t cached_bytes = 0;
static cudaStream_t stream = nullptr;
static bool initialized = false;
static std::mutex mtx;

__global__ void softmax_optimized_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float thread_max = -INFINITY;
    for (int col = tid; col < cols; col += block_dim) {
        thread_max = fmaxf(thread_max, row_input[col]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = block_dim / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    
    // Warp-level редукция без синхронизации для последних 32 потоков
    volatile float* v_sdata = sdata;
    if (tid < 32) {
        if (block_dim >= 64) v_sdata[tid] = fmaxf(v_sdata[tid], v_sdata[tid + 32]);
        if (block_dim >= 32) v_sdata[tid] = fmaxf(v_sdata[tid], v_sdata[tid + 16]);
        if (block_dim >= 16) v_sdata[tid] = fmaxf(v_sdata[tid], v_sdata[tid + 8]);
        if (block_dim >= 8)  v_sdata[tid] = fmaxf(v_sdata[tid], v_sdata[tid + 4]);
        if (block_dim >= 4)  v_sdata[tid] = fmaxf(v_sdata[tid], v_sdata[tid + 2]);
        if (block_dim >= 2)  v_sdata[tid] = fmaxf(v_sdata[tid], v_sdata[tid + 1]);
    }
    __syncthreads();
    float row_max = v_sdata[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += block_dim) {
        thread_sum += __expf(row_input[col] - row_max);
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = block_dim / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        if (block_dim >= 64) v_sdata[tid] += v_sdata[tid + 32];
        if (block_dim >= 32) v_sdata[tid] += v_sdata[tid + 16];
        if (block_dim >= 16) v_sdata[tid] += v_sdata[tid + 8];
        if (block_dim >= 8)  v_sdata[tid] += v_sdata[tid + 4];
        if (block_dim >= 4)  v_sdata[tid] += v_sdata[tid + 2];
        if (block_dim >= 2)  v_sdata[tid] += v_sdata[tid + 1];
    }
    __syncthreads();
    float row_sum = v_sdata[0];
    __syncthreads();

    float inv_sum = __fdividef(1.0f, row_sum);
    for (int col = tid; col < cols; col += block_dim) {
        row_output[col] = __expf(row_input[col] - row_max) * inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int size = input.size();
    int cols = size / row_count;
    size_t bytes = size * sizeof(float);
    std::vector<float> output(size);
    if (size == 0) return output;

    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized) {
        cudaStreamCreate(&stream);
        initialized = true;
    }

    if (bytes > cached_bytes) {
        if (d_in_cache) cudaFree(d_in_cache);
        if (d_out_cache) cudaFree(d_out_cache);
        cudaMalloc(&d_in_cache, bytes);
        cudaMalloc(&d_out_cache, bytes);
        cached_bytes = bytes;
    }

    cudaMemcpyAsync(d_in_cache, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    int threads_per_block = 1024;
    softmax_optimized_kernel<<<row_count, threads_per_block, threads_per_block * sizeof(float), stream>>>(d_in_cache, d_out_cache, cols);

    cudaMemcpyAsync(output.data(), d_out_cache, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return output;
}