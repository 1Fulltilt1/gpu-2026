#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

// Размеры блока и тайлов для оптимизации доступа к глобальной и разделяемой памяти
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

// CUDA-ядро для оптимизированного умножения матриц с использованием Shared Memory и Register Tiling
__global__ void __launch_bounds__(256) gemm_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) 
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // от 0 до 255 (256 потоков)

    // Разделяемая память для тайлов матриц A и B (с транспонированием A для избежания bank conflicts)
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    int threadRow = tid / (BN / TN); // 256 / 16 = 16 строк на поток
    int threadCol = tid % (BN / TN); // 16 колонок

    // Смещение указателей на начало текущего блока
    int globalRowStart = by * BM;
    int globalColStart = bx * BN;

    // Аккумуляторы в регистрах потока для подсчета блока TM x TN
    float accum[TM][TN] = {0.0f};

    // Цикл по всем тайлам вдоль измерения K
    for (int bk = 0; bk < N; bk += BK) {
        // Кооперативная загрузка тайла матрицы A из глобальной памяти в sA с использованием float4
        int a_inner_row = tid / 2;     // 128 строк
        int a_inner_col = (tid % 2) * 4; // столбцы по 4 элемента

        if ((globalRowStart + a_inner_row) < N && (bk + a_inner_col + 3) < N) {
            float4 a_vec = *reinterpret_cast<const float4*>(&A[(globalRowStart + a_inner_row) * N + (bk + a_inner_col)]);
            sA[a_inner_col + 0][a_inner_row] = a_vec.x;
            sA[a_inner_col + 1][a_inner_row] = a_vec.y;
            sA[a_inner_col + 2][a_inner_row] = a_vec.z;
            sA[a_inner_col + 3][a_inner_row] = a_vec.w;
        } else {
            // Безопасная обработка границ
            for (int i = 0; i < 4; ++i) {
                int r = globalRowStart + a_inner_row;
                int c = bk + a_inner_col + i;
                sA[a_inner_col + i][a_inner_row] = (r < N && c < N) ? A[r * N + c] : 0.0f;
            }
        }

        // Кооперативная загрузка тайла матрицы B из глобальной памяти в sB с использованием float4
        int b_inner_row = tid / 16;     // 16 строк
        int b_inner_col = (tid % 16) * 4; // столбцы

        if (b_inner_row < BK && (globalColStart + b_inner_col + 3) < N && (bk + b_inner_row) < N) {
            float4 b_vec = *reinterpret_cast<const float4*>(&B[(bk + b_inner_row) * N + (globalColStart + b_inner_col)]);
            *reinterpret_cast<float4*>(&sB[b_inner_row][b_inner_col]) = b_vec;
        } else {
            for (int i = 0; i < 4; ++i) {
                int r = bk + b_inner_row;
                int c = globalColStart + b_inner_col + i;
                if (b_inner_row < BK) {
                    sB[b_inner_row][b_inner_col + i] = (r < N && c < N) ? B[r * N + c] : 0.0f;
                }
            }
        }

        // Синхронизация потоков, чтобы убедиться, что весь тайл загружен в разделяемую память
        __syncthreads();

        // Вычисление произведения тайлов с разверткой циклов (#pragma unroll) и FMA инструкциями
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[k][threadRow * TM + i];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = sB[k][threadCol * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] = fmaf(regA[i], regB[j], accum[i][j]);
                }
            }
        }

        // Синхронизация перед переходом к загрузке следующего тайла
        __syncthreads();
    }

    // Запись результатов из регистров обратно в глобальную память матрицы C с проверкой границ
    int outRowBase = globalRowStart + threadRow * TM;
    int outColBase = globalColStart + threadCol * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = outRowBase + i;
            int col = outColBase + j;
            if (row < N && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}

// Основная функция-обертка для вызова из хост-кода (CPU)
std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = static_cast<size_t>(n) * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size, 0.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // Выделение памяти на GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Закрепление хост-памяти (Pinned Memory) для ускорения асинхронных пересылок PCI-e
    cudaHostRegister((void*)a.data(), bytes, cudaHostRegisterReadOnly);
    cudaHostRegister((void*)b.data(), bytes, cudaHostRegisterReadOnly);
    cudaHostRegister((void*)c.data(), bytes, cudaHostRegisterDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Асинхронная передача данных с хоста на устройство
    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    // Конфигурация запуска сетки CUDA (блок 16x16 потоков = 256 потоков)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + BN - 1) / BN, (n + BM - 1) / BM);

    // Запуск ядра в созданном стриме
    gemm_optimized_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, n);

    // Асинхронное копирование результата обратно на хост
    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    // Ожидание завершения всех операций в стриме
    cudaStreamSynchronize(stream);

    // Очистка ресурсов
    cudaStreamDestroy(stream);
    cudaHostUnregister((void*)a.data());
    cudaHostUnregister((void*)b.data());
    cudaHostUnregister((void*)c.data());

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}