import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

_KERNEL = r"""
extern "C" __global__ void layernorm_kernel(
        const float *input,
        const float *gamma,
        const float *beta,
              float *output,
              int    row_size,
              float  epsilon)
{
    extern __shared__ float shared_reduction[];  // [sum | sum_of_squares]

    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;
    const int row         = blockIdx.x;

    const float *input_row = input  + (long long)row * row_size;
    float *output_row      = output + (long long)row * row_size;

    float partial_sum         = 0.f;
    float partial_sum_squares = 0.f;
    for (int col = tid; col < row_size; col += num_threads) {
        const float value = input_row[col];
        partial_sum         += value;
        partial_sum_squares += value * value;
    }

    shared_reduction[tid]               = partial_sum;
    shared_reduction[tid + num_threads] = partial_sum_squares;
    __syncthreads();

    for (int stride = num_threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_reduction[tid]               += shared_reduction[tid + stride];
            shared_reduction[tid + num_threads] += shared_reduction[tid + num_threads + stride];
        }
        __syncthreads();
    }

    const float mean            = shared_reduction[0]               / row_size;
    const float mean_of_squares = shared_reduction[num_threads]     / row_size;
    const float inverse_std     = rsqrtf(mean_of_squares - mean * mean + epsilon);

    for (int col = tid; col < row_size; col += num_threads) {
        output_row[col] = gamma[col] * (input_row[col] - mean) * inverse_std + beta[col];
    }
}
"""

_module              = SourceModule(_KERNEL, no_extern_c=True)
_layernorm_kernel    = _module.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row-wise
        normalized results.
    """
    input_flat = np.asarray(input, dtype=np.float32).ravel()
    gamma_flat = np.asarray(gamma, dtype=np.float32).ravel()
    beta_flat  = np.asarray(beta,  dtype=np.float32).ravel()

    num_features = int(row_size)
    num_rows     = input_flat.size // num_features
    output_flat  = np.empty_like(input_flat)

    threads_per_block = min(1024, num_features)
    shared_bytes      = 2 * threads_per_block * np.dtype(np.float32).itemsize

    _layernorm_kernel(
        cuda.In(input_flat),
        cuda.In(gamma_flat),
        cuda.In(beta_flat),
        cuda.Out(output_flat),
        np.int32(num_features),
        np.float32(eps),
        block=(threads_per_block, 1, 1),
        grid=(num_rows, 1, 1),
        shared=shared_bytes,
    )
    return output_flat
