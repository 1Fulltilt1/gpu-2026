#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

constexpr float kSqrtTwoOverPi = 0.7978845608f;
constexpr float kGeluCoeff = 0.044715f;

__global__ void GeluKernel(const float4* __restrict__ input,
                           float4* __restrict__ output,
                           int vec4_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vec4_count) {
        float4 x = input[idx];

        float t;

        t = kSqrtTwoOverPi * (x.x + kGeluCoeff * x.x * x.x * x.x);
        float e = __expf(2.0f * t);
        x.x *= 0.5f * (1.0f + (e - 1.0f) / (e + 1.0f));

        t = kSqrtTwoOverPi * (x.y + kGeluCoeff * x.y * x.y * x.y);
        e = __expf(2.0f * t);
        x.y *= 0.5f * (1.0f + (e - 1.0f) / (e + 1.0f));

        t = kSqrtTwoOverPi * (x.z + kGeluCoeff * x.z * x.z * x.z);
        e = __expf(2.0f * t);
        x.z *= 0.5f * (1.0f + (e - 1.0f) / (e + 1.0f));

        t = kSqrtTwoOverPi * (x.w + kGeluCoeff * x.w * x.w * x.w);
        e = __expf(2.0f * t);
        x.w *= 0.5f * (1.0f + (e - 1.0f) / (e + 1.0f));

        output[idx] = x;
    }
}

float* GetDeviceBuffer(float*& buffer, int required_floats) {
    static int allocated_floats = 0;

    if (required_floats > allocated_floats) {
        if (buffer != nullptr) {
            cudaFree(buffer);
        }

        cudaMalloc(&buffer, required_floats * sizeof(float));
        allocated_floats = required_floats;
    }

    return buffer;
}

float* GetInputBuffer(int size) {
    static float* buffer = nullptr;
    return GetDeviceBuffer(buffer, size);
}

float* GetOutputBuffer(int size) {
    static float* buffer = nullptr;
    return GetDeviceBuffer(buffer, size);
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int size = static_cast<int>(input.size());
    std::vector<float> output(size);

    if (size == 0) {
        return output;
    }

    const int vec4_count = (size + 3) / 4;
    const int padded_size = vec4_count * 4;

    float* d_input = GetInputBuffer(padded_size);
    float* d_output = GetOutputBuffer(padded_size);

    if (size != padded_size) {
        std::vector<float> padded_input(input);
        padded_input.resize(padded_size, 0.0f);

        cudaMemcpy(d_input, padded_input.data(),
                   padded_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(d_input, input.data(),
                   size * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    const int block_size = 256;
    const int grid_size = (vec4_count + block_size - 1) / block_size;

    GeluKernel<<<grid_size, block_size>>>(
        reinterpret_cast<const float4*>(d_input),
        reinterpret_cast<float4*>(d_output),
        vec4_count);

    cudaMemcpy(output.data(), d_output,
               size * sizeof(float),
               cudaMemcpyDeviceToHost);

    return output;
}
