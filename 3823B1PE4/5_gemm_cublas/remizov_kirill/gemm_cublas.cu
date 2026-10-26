#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) return {};
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (a.size() < count || b.size() < count) return {};
    const std::size_t bytes = count * sizeof(float);

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

    cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // cuBLAS works with column-major matrices: a row-major matrix X is seen as
    // X^T. Therefore row-major C = A * B is the same as column-major
    // C^T = B^T * A^T, so passing B and then A avoids any explicit transpose.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                &alpha, dev_b, n, dev_a, n, &beta, dev_c, n);

    std::vector<float> result(count);
    cudaMemcpy(result.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cublasDestroy(handle);
    return result;
}
