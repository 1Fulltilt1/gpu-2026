#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

    constexpr char kSource[] = R"CLC(
__kernel void gelu_vec4(__global const float4* input,
                        __global float4* output,
                        const uint n4,
                        __global const float* tail_in,
                        __global float* tail_out,
                        const uint tail_n) {
    uint i = get_global_id(0);
    if (i < n4) {
        float4 x = input[i];
        float4 x3 = x * x * x;
        float4 z = (float4)(0.7978845608f) * (x + (float4)(0.044715f) * x3);
        float4 e = exp((float4)(-2.0f) * z);
        output[i] = x / ((float4)(1.0f) + e);
    }
    uint t = get_global_id(0) - n4;
    if (get_global_id(0) >= n4 && t < tail_n) {
        float x = tail_in[t];
        float x3 = x * x * x;
        float z = 0.7978845608f * (x + 0.044715f * x3);
        tail_out[t] = x / (1.0f + exp(-2.0f * z));
    }
}
)CLC";

    struct Ctx {
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
        cl_device_id device = nullptr;
        size_t max_wg = 256;

        cl_mem in_buf = nullptr;
        cl_mem out_buf = nullptr;
        cl_mem tail_in = nullptr;
        cl_mem tail_out = nullptr;
        size_t capacity_bytes = 0;
        size_t tail_capacity = 0;

        ~Ctx() {
            if (in_buf) clReleaseMemObject(in_buf);
            if (out_buf) clReleaseMemObject(out_buf);
            if (tail_in) clReleaseMemObject(tail_in);
            if (tail_out) clReleaseMemObject(tail_out);
            if (kernel) clReleaseKernel(kernel);
            if (program) clReleaseProgram(program);
            if (queue) clReleaseCommandQueue(queue);
            if (context) clReleaseContext(context);
        }
    };

    std::mutex g_init_mutex;
    std::unordered_map<int, Ctx*> g_cache;

    Ctx* GetOrCreate(int platform) {
        std::lock_guard<std::mutex> lock(g_init_mutex);
        auto it = g_cache.find(platform);
        if (it != g_cache.end()) return it->second;

        cl_int err = CL_SUCCESS;

        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0)
            throw std::runtime_error("OpenCL: no platforms");
        if (platform < 0 || static_cast<cl_uint>(platform) >= num_platforms)
            throw std::runtime_error("OpenCL: platform index out of range");

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clGetPlatformIDs failed");

        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0)
            throw std::runtime_error("OpenCL: no GPU device on platform");

        cl_device_id device = nullptr;
        err = clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clGetDeviceIDs failed");

        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platforms[platform]),
            0
        };
        cl_context context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clCreateContext failed");

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            throw std::runtime_error("OpenCL: clCreateCommandQueue failed");
        }

        const char* src = kSource;
        cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            throw std::runtime_error("OpenCL: clCreateProgramWithSource failed");
        }

        err = clBuildProgram(program, 1, &device,
            "-cl-fast-relaxed-math -cl-mad-enable",
            nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            throw std::runtime_error("OpenCL build failed:\n" + log);
        }

        cl_kernel kernel = clCreateKernel(program, "gelu_vec4", &err);
        if (err != CL_SUCCESS) {
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            throw std::runtime_error("OpenCL: clCreateKernel failed");
        }

        size_t max_wg = 256;
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(max_wg), &max_wg, nullptr);

        Ctx* c = new Ctx();
        c->context = context;
        c->queue = queue;
        c->program = program;
        c->kernel = kernel;
        c->device = device;
        c->max_wg = max_wg ? max_wg : 256;

        g_cache[platform] = c;
        return c;
    }

    void EnsureCapacity(Ctx* c, size_t n) {
        const size_t main_n = n & ~size_t(3);
        const size_t tail_n = n - main_n;
        const size_t main_bytes = main_n * sizeof(float);

        if (main_bytes > c->capacity_bytes) {
            if (c->in_buf)  clReleaseMemObject(c->in_buf);
            if (c->out_buf) clReleaseMemObject(c->out_buf);
            cl_int err = CL_SUCCESS;
            c->in_buf = clCreateBuffer(c->context,
                CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                main_bytes, nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: in_buf alloc failed");
            c->out_buf = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY,
                main_bytes, nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: out_buf alloc failed");
            c->capacity_bytes = main_bytes;
        }

        if (tail_n > c->tail_capacity) {
            if (c->tail_in)  clReleaseMemObject(c->tail_in);
            if (c->tail_out) clReleaseMemObject(c->tail_out);
            cl_int err = CL_SUCCESS;
            c->tail_in = clCreateBuffer(c->context, CL_MEM_READ_ONLY,
                tail_n * sizeof(float), nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: tail_in alloc failed");
            c->tail_out = clCreateBuffer(c->context, CL_MEM_WRITE_ONLY,
                tail_n * sizeof(float), nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: tail_out alloc failed");
            c->tail_capacity = tail_n;
        }
    }

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) return {};

    Ctx* c = GetOrCreate(platform);
    EnsureCapacity(c, n);

    const size_t main_n = n & ~size_t(3);
    const size_t tail_n = n - main_n;
    const size_t n4 = main_n / 4;

    if (main_n > 0) {
        void* host_in = clEnqueueMapBuffer(c->queue, c->in_buf, CL_TRUE,
            CL_MAP_WRITE, 0,
            main_n * sizeof(float),
            0, nullptr, nullptr, nullptr);
        if (!host_in) throw std::runtime_error("OpenCL: map in_buf failed");
        std::memcpy(host_in, input.data(), main_n * sizeof(float));
        clEnqueueUnmapMemObject(c->queue, c->in_buf, host_in, 0, nullptr, nullptr);
    }

    if (tail_n > 0) {
        clEnqueueWriteBuffer(c->queue, c->tail_in, CL_FALSE, 0,
            tail_n * sizeof(float),
            input.data() + main_n, 0, nullptr, nullptr);
    }

    cl_int err = CL_SUCCESS;
    err = clSetKernelArg(c->kernel, 0, sizeof(cl_mem), &c->in_buf);
    err |= clSetKernelArg(c->kernel, 1, sizeof(cl_mem), &c->out_buf);
    cl_uint n4_arg = static_cast<cl_uint>(n4);
    err |= clSetKernelArg(c->kernel, 2, sizeof(cl_uint), &n4_arg);
    err |= clSetKernelArg(c->kernel, 3, sizeof(cl_mem), &c->tail_in);
    err |= clSetKernelArg(c->kernel, 4, sizeof(cl_mem), &c->tail_out);
    cl_uint tail_arg = static_cast<cl_uint>(tail_n);
    err |= clSetKernelArg(c->kernel, 5, sizeof(cl_uint), &tail_arg);
    if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: clSetKernelArg failed");

    const size_t total_items = n4 + tail_n;
    const size_t local = std::min(c->max_wg, size_t(256));
    const size_t global = ((total_items + local - 1) / local) * local;

    err = clEnqueueNDRangeKernel(c->queue, c->kernel, 1, nullptr,
        &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: enqueue kernel failed");

    std::vector<float> output(n);

    cl_event ev_main = nullptr;
    if (main_n > 0) {
        err = clEnqueueReadBuffer(c->queue, c->out_buf, CL_FALSE, 0,
            main_n * sizeof(float),
            output.data(), 0, nullptr, &ev_main);
        if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: read main failed");
    }
    if (tail_n > 0) {
        err = clEnqueueReadBuffer(c->queue, c->tail_out, CL_FALSE, 0,
            tail_n * sizeof(float),
            output.data() + main_n, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("OpenCL: read tail failed");
    }

    clFinish(c->queue);
    if (ev_main) clReleaseEvent(ev_main);

    return output;
}