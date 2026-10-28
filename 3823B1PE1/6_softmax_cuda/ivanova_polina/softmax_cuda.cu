#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <algorithm>

static float* d_in_cache = nullptr;
static float* d_out_cache = nullptr;
static size_t cached_bytes = 0;
static cudaStream_t stream = nullptr;
static bool initialized = false;
static std::mutex mtx;

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(256) softmax_robust_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    __shared__ float s_reduce[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;
    int lane = tid % 32;
    int wid = tid / 32;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += block_dim) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    float warp_max = warpReduceMax(local_max);
    if (lane == 0) {
        s_reduce[wid] = warp_max;
    }
    __syncthreads();

    float block_max = (tid < 32) ? s_reduce[lane] : -INFINITY;
    if (wid == 0) {
        block_max = warpReduceMax(block_max);
        if (lane == 0) {
            s_reduce[0] = block_max;
        }
    }
    __syncthreads();
    float row_max = s_reduce[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_dim) {
        local_sum += __expf(row_in[i] - row_max);
    }

    float warp_sum = warpReduceSum(local_sum);
    if (lane == 0) {
        s_reduce[wid] = warp_sum;
    }
    __syncthreads();

    float block_sum = (tid < 32) ? s_reduce[lane] : 0.0f;
    if (wid == 0) {
        block_sum = warpReduceSum(block_sum);
        if (lane == 0) {
            s_reduce[0] = block_sum;
        }
    }
    __syncthreads();
    float row_sum = s_reduce[0];

    float inv_sum = __fdividef(1.0f, row_sum);

    for (int i = tid; i < cols; i += block_dim) {
        row_out[i] = __expf(row_in[i] - row_max) * inv_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int size = input.size();
    if (size == 0) return {};
    int cols = size / row_count;
    size_t bytes = size * sizeof(float);
    std::vector<float> output(size);

    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized) {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        initialized = true;
    }

    if (bytes > cached_bytes) {
        if (d_in_cache) cudaFree(d_in_cache);
        if (d_out_cache) cudaFree(d_out_cache);
        cudaMalloc(&d_in_cache, bytes);
        cudaMalloc(&d_out_cache, bytes);
        cached_bytes = bytes;
    }

    cudaHostRegister(const_cast<float*>(input.data()), bytes, cudaHostRegisterDefault);
    cudaHostRegister(output.data(), bytes, cudaHostRegisterDefault);

    cudaMemcpyAsync(d_in_cache, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    int threads_per_block = 256;
    softmax_robust_kernel<<<row_count, threads_per_block, 0, stream>>>(d_in_cache, d_out_cache, cols);

    cudaMemcpyAsync(output.data(), d_out_cache, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaHostUnregister(const_cast<float*>(input.data()));
    cudaHostUnregister(output.data());

    return output;
}