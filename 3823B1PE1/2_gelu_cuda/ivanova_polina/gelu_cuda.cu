#include "gelu_cuda_opt.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void gelu_vectorized_kernel(const float4* __restrict__ input, float4* __restrict__ output, size_t n_v4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_v4) {
        float4 in_val = input[idx];
        float4 out_val;
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        // Обработка 1-го элемента во float4
        {
            float x = in_val.x;
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            float exp_2y = __expf(2.0f * inner);
            float fast_tanh = (exp_2y - 1.0f) / (exp_2y + 1.0f);
            out_val.x = 0.5f * x * (1.0f + fast_tanh);
        }
        // Обработка 2-го элемента
        {
            float x = in_val.y;
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            float exp_2y = __expf(2.0f * inner);
            float fast_tanh = (exp_2y - 1.0f) / (exp_2y + 1.0f);
            out_val.y = 0.5f * x * (1.0f + fast_tanh);
        }
        // Обработка 3-го элемента
        {
            float x = in_val.z;
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            float exp_2y = __expf(2.0f * inner);
            float fast_tanh = (exp_2y - 1.0f) / (exp_2y + 1.0f);
            out_val.z = 0.5f * x * (1.0f + fast_tanh);
        }
        // Обработка 4-го элемента
        {
            float x = in_val.w;
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            float exp_2y = __expf(2.0f * inner);
            float fast_tanh = (exp_2y - 1.0f) / (exp_2y + 1.0f);
            out_val.w = 0.5f * x * (1.0f + fast_tanh);
        }
        output[idx] = out_val;
    }
}

__global__ void gelu_scalar_kernel(const float* __restrict__ input, float* __restrict__ output, size_t start_idx, size_t n) {
    size_t idx = start_idx + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        float exp_2y = __expf(2.0f * inner);
        float fast_tanh = (exp_2y - 1.0f) / (exp_2y + 1.0f);
        output[idx] = 0.5f * x * (1.0f + fast_tanh);
    }
}

std::vector<float> GeluCUDAOpt(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    float* d_in = nullptr;
    float* d_out = nullptr;
    size_t bytes = n * sizeof(float);

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    size_t n_v4 = n / 4;
    size_t rem = n % 4;

    if (n_v4 > 0) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n_v4 + threadsPerBlock - 1) / threadsPerBlock;
        gelu_vectorized_kernel<<<blocksPerGrid, threadsPerBlock>>>((const float4*)d_in, (float4*)d_out, n_v4);
    }
    if (rem > 0) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (rem + threadsPerBlock - 1) / threadsPerBlock;
        gelu_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n - rem, n);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    return output;
}