#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

// Функция принимает матрицы a и b размера n x n, представленные в виде одномерных векторов,
// и возвращает результирующий вектор c размера n x n.
std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __NAIVE_GEMM_CUDA_H