import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

_KERNEL = r"""
extern "C" __global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps)
{
    const int row = blockIdx.x;
    const float* row_in  = input  + (long long)row * row_size;
    float*       row_out = output + (long long)row * row_size;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < row_size; j += blockDim.x) {
        local_sum += row_in[j];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float mean = sdata[0] / (float)row_size;

    float local_var = 0.0f;
    for (int j = threadIdx.x; j < row_size; j += blockDim.x) {
        const float d = row_in[j] - mean;
        local_var += d * d;
    }
    sdata[threadIdx.x] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float var = sdata[0] / (float)row_size;
    const float inv_std = rsqrtf(var + eps);

    for (int j = threadIdx.x; j < row_size; j += blockDim.x) {
        row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}
"""

_module = None
_kernel = None


def _get_kernel():
    global _module, _kernel
    if _kernel is None:
        _module = SourceModule(_KERNEL)
        _kernel = _module.get_function("layernorm_kernel")
    return _kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.asarray(input, dtype=np.float32).ravel()
    g = np.asarray(gamma, dtype=np.float32).ravel()
    b = np.asarray(beta,  dtype=np.float32).ravel()

    if x.size % row_size != 0:
        raise ValueError("input size must be divisible by row_size")
    if g.size != row_size or b.size != row_size:
        raise ValueError("gamma and beta must have length row_size")

    row_count = x.size // row_size
    out = np.empty_like(x)

    x_gpu = cuda.mem_alloc(x.nbytes)
    g_gpu = cuda.mem_alloc(g.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    o_gpu = cuda.mem_alloc(out.nbytes)

    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(g_gpu, g)
    cuda.memcpy_htod(b_gpu, b)

    kernel = _get_kernel()

    threads = 128
    blocks = row_count
    shmem = threads * 4

    kernel(x_gpu, g_gpu, b_gpu, o_gpu,
           np.int32(row_size), np.float32(eps),
           block=(threads, 1, 1),
           grid=(blocks, 1, 1),
           shared=shmem)

    cuda.memcpy_dtoh(out, o_gpu)

    x_gpu.free(); g_gpu.free(); b_gpu.free(); o_gpu.free()

    return out