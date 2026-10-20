#include "gelu_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608f;
constexpr float kGeluCoef = 0.044715f;

__global__ void GeluKernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = input[idx];

        float t = kSqrt2OverPi * (x + kGeluCoef * x * x * x);

        float e = __expf(2.0f * t);
        float tanh_t = (e - 1.0f) / (e + 1.0f);

        output[idx] = 0.5f * x * (1.0f + tanh_t);
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = static_cast<int>(input.size());
    std::vector<float> output(size);

    if (size == 0) {
        return output;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input.data(), size * sizeof(float),
               cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    GeluKernel<<<grid_size, block_size>>>(d_input, d_output, size);

    cudaMemcpy(output.data(), d_output, size * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}