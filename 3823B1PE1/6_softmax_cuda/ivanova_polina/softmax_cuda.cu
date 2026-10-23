#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

// Для редукции максимума и суммы экспонент используем разделяемая память
__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int cols) {

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

    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += block_dim) {
        thread_sum += expf(row_input[col] - row_max);
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int col = tid; col < cols; col += block_dim) {
        row_output[col] = expf(row_input[col] - row_max) / row_sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    int size = input.size();
    int cols = size / row_count;
    std::vector<float> output(size);

    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Используем 1024 потока на блок, так как размер строки - 16384 - гарантированно больше и является степенью двойки
    int threads_per_block = 1024;
    int num_blocks = row_count;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    softmax_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(d_input, d_output, cols);

    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}