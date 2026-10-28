#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char* kGeluVectorKernelSource = R"(
__kernel void gelu_vector_kernel(__global const float4* input, __global float4* output, const int num_vecs) {
    int idx = get_global_id(0);
    if (idx < num_vecs) {
        float4 x = input[idx];
        float4 inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        float4 exp_val = native_exp(2.0f * inner);
        float4 t = (exp_val - 1.0f) / (exp_val + 1.0f);
        output[idx] = 0.5f * x * (1.0f + t);
    }
}
)";

struct FastOpenCLWorkspace {
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    
    cl_mem d_input = nullptr;
    cl_mem d_output = nullptr;
    size_t capacity_bytes = 0;
    bool initialized = false;

    void init(int platform_index, size_t required_bytes) {
        if (!initialized) {
            cl_uint num_platforms = 0;
            clGetPlatformIDs(0, nullptr, &num_platforms);
            if (num_platforms == 0) throw std::runtime_error("No OpenCL platforms found.");

            std::vector<cl_platform_id> platforms(num_platforms);
            clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
            platform_id = platforms[platform_index];

            cl_device_id device;
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            device_id = device;

            context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, nullptr);
#ifdef CL_VERSION_2_0
            queue = clCreateCommandQueueWithProperties(context, device_id, 0, nullptr);
#else
            queue = clCreateCommandQueue(context, device_id, 0, nullptr);
#endif
            program = clCreateProgramWithSource(context, 1, &kGeluVectorKernelSource, nullptr, nullptr);
            clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math", nullptr, nullptr);
            
            kernel = clCreateKernel(program, "gelu_vector_kernel", nullptr);
            initialized = true;
        }

        if (capacity_bytes < required_bytes) {
            if (d_input) clReleaseMemObject(d_input);
            if (d_output) clReleaseMemObject(d_output);

            d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, required_bytes, nullptr, nullptr);
            d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, required_bytes, nullptr, nullptr);
            capacity_bytes = required_bytes;
        }
    }
};

static FastOpenCLWorkspace g_fast_ws;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    size_t n = input.size();
    size_t bytes = n * sizeof(float);

    g_fast_ws.init(platform, bytes);

    clEnqueueWriteBuffer(g_fast_ws.queue, g_fast_ws.d_input, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    int num_vecs = static_cast<int>(n / 4);
    clSetKernelArg(g_fast_ws.kernel, 0, sizeof(cl_mem), &g_fast_ws.d_input);
    clSetKernelArg(g_fast_ws.kernel, 1, sizeof(cl_mem), &g_fast_ws.d_output);
    clSetKernelArg(g_fast_ws.kernel, 2, sizeof(int), &num_vecs);

    size_t local_work_size = 256;
    size_t global_work_size = ((num_vecs + local_work_size - 1) / local_work_size) * local_work_size;

    clEnqueueNDRangeKernel(g_fast_ws.queue, g_fast_ws.kernel, 1, nullptr, 
                           &global_work_size, &local_work_size, 0, nullptr, nullptr);

    std::vector<float> output(n);

    clEnqueueReadBuffer(g_fast_ws.queue, g_fast_ws.d_output, CL_TRUE, 0, 
                        bytes, output.data(), 0, nullptr, nullptr);

    return output;
}