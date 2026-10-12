%%writefile softmax_cuda.cu
#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int THREADS = 256;
constexpr int NWARPS  = THREADS / 32;

__device__ __forceinline__ float warpReduceMax(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    return v;
}

__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void SoftmaxKernel(const float* __restrict__ in,
                              float* __restrict__ out,
                              int row_size)
{
    __shared__ float s_warp_max[NWARPS];
    __shared__ float s_warp_sum[NWARPS];
    __shared__ float s_row_max;
    __shared__ float s_row_sum;

    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    const float* row_in  = in  + (std::size_t)row * row_size;
    float*       row_out = out + (std::size_t)row * row_size;

    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x)
        local_max = fmaxf(local_max, row_in[i]);

    local_max = warpReduceMax(local_max);
    if (lane == 0) s_warp_max[warp] = local_max;
    __syncthreads();

    if (tid == 0) {
        float m = s_warp_max[0];
        #pragma unroll
        for (int i = 1; i < NWARPS; ++i) m = fmaxf(m, s_warp_max[i]);
        s_row_max = m;
    }
    __syncthreads();
    const float row_max = s_row_max;

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x)
        local_sum += __expf(row_in[i] - row_max);

    local_sum = warpReduceSum(local_sum);
    if (lane == 0) s_warp_sum[warp] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float s = s_warp_sum[0];
        #pragma unroll
        for (int i = 1; i < NWARPS; ++i) s += s_warp_sum[i];
        s_row_sum = s;
    }
    __syncthreads();
    const float inv_sum = 1.0f / s_row_sum;

    for (int i = tid; i < row_size; i += blockDim.x)
        row_out[i] = __expf(row_in[i] - row_max) * inv_sum;
}

float*      g_d_in  = nullptr;
float*      g_d_out = nullptr;
std::size_t g_cap   = 0;

void ensureBuffers(std::size_t elems) {
    if (elems <= g_cap) return;
    if (g_d_in)  cudaFree(g_d_in);
    if (g_d_out) cudaFree(g_d_out);
    const std::size_t bytes = elems * sizeof(float);
    cudaMalloc(&g_d_in,  bytes);
    cudaMalloc(&g_d_out, bytes);
    g_cap = elems;
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count)
{
    const int total = (int)input.size();
    std::vector<float> output(total);
    if (total == 0 || row_count <= 0) return output;

    const int row_size = total / row_count;
    const std::size_t bytes = (std::size_t)total * sizeof(float);
    ensureBuffers((std::size_t)total);

    cudaMemcpyAsync(g_d_in, input.data(), bytes, cudaMemcpyHostToDevice, 0);
    SoftmaxKernel<<<row_count, THREADS>>>(g_d_in, g_d_out, row_size);
    cudaMemcpyAsync(output.data(), g_d_out, bytes, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize();
    return output;
}