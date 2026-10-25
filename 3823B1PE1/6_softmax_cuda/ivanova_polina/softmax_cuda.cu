# include "softmax_cuda.h"
# include <cuda_runtime.h>
# include <cmath>
# include <mutex>

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

__global__ void softmax_optimized_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {
    __shared__ float shared_scratch[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int block_dim = blockDim.x;

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float thread_max = -INFINITY;

    if (cols % 4 == 0) {
        const float4* row_input_v4 = reinterpret_cast<const float4*>(row_input);
        int cols_v4 = cols / 4;
        for (int col = tid; col < cols_v4; col += block_dim) {
            float4 val = row_input_v4[col];
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
        }
    } else {
        for (int col = tid; col < cols; col += block_dim) {
            thread_max = fmaxf(thread_max, row_input[col]);
        }
    }

    float warp_max = warpReduceMax(thread_max);
    if (lane == 0) {
        shared_scratch[wid] = warp_max;
    }
    __syncthreads();

    float block_max = (tid < 32) ? shared_scratch[lane] : -INFINITY;
    if (wid == 0) {
        block_max = warpReduceMax(block_max);
        if (lane == 0) {
            shared_scratch[0] = block_max;
        }
    }
    __syncthreads();
    float row_max = shared_scratch[0];

    float thread_sum = 0.0f;
    if (cols % 4 == 0) {
        const float4* row_input_v4 = reinterpret_cast<const float4*>(row_input);
        int cols_v4 = cols / 4;
        for (int col = tid; col < cols_v4; col += block_dim) {
            float4 val = row_input_v4[col];
            thread_sum += __expf(val.x - row_max) +
                          __expf(val.y - row_max) +
                          __expf(val.z - row_max) +
                          __expf(val.w - row_max);
        }
    } else {
        for (int col = tid; col < cols; col += block_dim) {
            thread_sum += __expf(row_input[col] - row_max);
        }
    }

    float warp_sum = warpReduceSum(thread_sum);
    if (lane == 0) {
        shared_scratch[wid] = warp_sum;
    }
    __syncthreads();

    float block_sum = (tid < 32) ? shared_scratch[lane] : 0.0f;
    if (wid == 0) {
        block_sum = warpReduceSum(block_sum);
        if (lane == 0) {
            shared_scratch[0] = block_sum;
        }
    }
    __syncthreads();
    float row_sum = shared_scratch[0];

    float inv_sum = __fdividef(1.0f, row_sum);
    if (cols % 4 == 0) {
        const float4* row_input_v4 = reinterpret_cast<const float4*>(row_input);
        float4* row_output_v4 = reinterpret_cast<float4*>(row_output);
        int cols_v4 = cols / 4;
        for (int col = tid; col < cols_v4; col += block_dim) {
            float4 val = row_input_v4[col];
            float4 out_val;
            out_val.x = __expf(val.x - row_max) * inv_sum;
            out_val.y = __expf(val.y - row_max) * inv_sum;
            out_val.z = __expf(val.z - row_max) * inv_sum;
            out_val.w = __expf(val.w - row_max) * inv_sum;
            row_output_v4[col] = out_val;
        }
    } else {
        for (int col = tid; col < cols; col += block_dim) {
            row_output[col] = __expf(row_input[col] - row_max) * inv_sum;
        }
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

    float* h_in_ptr = const_cast<float*>(input.data());
    float* h_out_ptr = output.data();

    cudaHostRegister(h_in_ptr, bytes, cudaHostRegisterDefault);
    cudaHostRegister(h_out_ptr, bytes, cudaHostRegisterDefault);

    cudaMemcpyAsync(d_in_cache, h_in_ptr, bytes, cudaMemcpyHostToDevice, stream);

    int threads_per_block = 1024;
    softmax_optimized_kernel<<<row_count, threads_per_block, 0, stream>>>(d_in_cache, d_out_cache, cols);

    cudaMemcpyAsync(h_out_ptr, d_out_cache, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaHostUnregister(h_in_ptr);
    cudaHostUnregister(h_out_ptr);

    return output;
}