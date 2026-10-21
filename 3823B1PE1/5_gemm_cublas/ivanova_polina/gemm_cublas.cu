#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    int size = n * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size, 0.0f);

    if (n == 0) return c;

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaHostRegister((void*)a.data(), bytes, cudaHostRegisterDefault);
    cudaHostRegister((void*)b.data(), bytes, cudaHostRegisterDefault);
    cudaHostRegister((void*)c.data(), bytes, cudaHostRegisterDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Меняем местами матрицы B и A, оставляя CUBLAS_OP_N
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaHostUnregister((void*)a.data());
    cudaHostUnregister((void*)b.data());
    cudaHostUnregister((void*)c.data());

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}