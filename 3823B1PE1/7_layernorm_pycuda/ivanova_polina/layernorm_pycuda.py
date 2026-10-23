import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Базовый (простой) CUDA-кёрнел на C++ для PyCUDA
_BASE_KERNEL = """
#define MAX_THREADS 1024

__global__ void base_layernorm_kernel(const float* input, const float* gamma, const float* beta, float* output, int row_size, float eps) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* row_in = input + row_idx * row_size;
    float* row_out = output + row_idx * row_size;

    __shared__ float s_mean[MAX_THREADS];
    __shared__ float s_var[MAX_THREADS];

    // Шаг 1: Вычисляем сумму элементов строки для среднего
    float thread_sum = 0.0f;
    for (int i = tid; i < row_size; i += num_threads) {
        thread_sum += row_in[i];
    }
    s_mean[tid] = thread_sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }
    float mean = s_mean[0] / row_size;

    // Шаг 2: Вычисляем дисперсию
    float thread_var_sum = 0.0f;
    for (int i = tid; i < row_size; i += num_threads) {
        float diff = row_in[i] - mean;
        thread_var_sum += diff * diff;
    }
    s_var[tid] = thread_var_sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_var[tid] += s_var[tid + s];
        }
        __syncthreads();
    }

    float var = s_var[0] / row_size;
    float inv_std = 1.0f / sqrtf(var + eps);

    // Шаг 3: Нормализация
    for (int i = tid; i < row_size; i += num_threads) {
        row_out[i] = ((row_in[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}
"""

_base_mod = SourceModule(_BASE_KERNEL)
_base_fn = _base_mod.get_function("base_layernorm_kernel")

def layernorm_pycuda(input_data, gamma, beta, row_size, eps=1e-5):
    """
    Базовая версия LayerNorm, переписанная на PyCUDA для предотвращения конфликта контекстов CUDA.
    """
    input_arr = np.ascontiguousarray(input_data, dtype=np.float32)
    gamma_arr = np.ascontiguousarray(gamma, dtype=np.float32)
    beta_arr = np.ascontiguousarray(beta, dtype=np.float32)
    
    num_elements = input_arr.size
    num_rows = num_elements // row_size
    output_arr = np.empty_like(input_arr)

    input_gpu = cuda.mem_alloc(input_arr.nbytes)
    gamma_gpu = cuda.mem_alloc(gamma_arr.nbytes)
    beta_gpu = cuda.mem_alloc(beta_arr.nbytes)
    output_gpu = cuda.mem_alloc(output_arr.nbytes)

    cuda.memcpy_htod(input_gpu, input_arr)
    cuda.memcpy_htod(gamma_gpu, gamma_arr)
    cuda.memcpy_htod(beta_gpu, beta_arr)

    threads_per_block = min(1024, row_size)
    if (threads_per_block & (threads_per_block - 1)) != 0:
        threads_per_block = 1 << (threads_per_block - 1).bit_length() - 1
        
    grid_size = (num_rows, 1, 1)
    block_size = (threads_per_block, 1, 1)

    _base_fn(
        input_gpu,
        gamma_gpu,
        beta_gpu,
        output_gpu,
        np.int32(row_size),
        np.float32(eps),
        block=block_size,
        grid=grid_size
    )

    cuda.memcpy_dtoh(output_arr, output_gpu)
    
    input_gpu.free()
    gamma_gpu.free()
    beta_gpu.free()
    output_gpu.free()
    
    return output_arr