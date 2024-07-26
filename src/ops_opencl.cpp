// ops_opencl.cpp
#include "ops.h"
#include <OpenCL/opencl.h>
#include <iostream>
#include <vector>

namespace ops {

const char* kernelSource = R"(
__kernel void add_kernel(__global const float* a, __global const float* b, __global float* result, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}
)";

void add_gpu(const Tensor& a, const Tensor& b, Tensor& result) {
    cl_int err;
    
    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting platform ID: " << err << std::endl;
        return;
    }

    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting device ID: " << err << std::endl;
        return;
    }

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context: " << err << std::endl;
        return;
    }

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating command queue: " << err << std::endl;
        return;
    }

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating program: " << err << std::endl;
        return;
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error building program: " << err << std::endl;
        return;
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "add_kernel", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel: " << err << std::endl;
        return;
    }

    int size = a.shape()[0];

    // Create buffers
    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), (void*)a.data(), &err);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), (void*)b.data(), &err);
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    // Execute kernel
    size_t global_work_size = size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel: " << err << std::endl;
        return;
    }

    // Read result
    err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, size * sizeof(float), result.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error reading result buffer: " << err << std::endl;
        return;
    }

    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(result_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

const char* matmul_kernel_source = R"(
__kernel void matmul_kernel(__global const float* a, __global const float* b, __global float* result,
                            const int m, const int n, const int k) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        result[row * n + col] = sum;
    }
}
)";


void matmul_gpu(const Tensor& a, const Tensor& b, Tensor& result) {
    cl_int err;
    
    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting platform ID: " << err << std::endl;
        return;
    }

    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting device ID: " << err << std::endl;
        return;
    }

    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context: " << err << std::endl;
        return;
    }

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating command queue: " << err << std::endl;
        return;
    }

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, &matmul_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating program: " << err << std::endl;
        return;
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error building program: " << err << std::endl;
        // Print build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        return;
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "matmul_kernel", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel: " << err << std::endl;
        return;
    }

    int m = a.shape()[0];
    int n = b.shape()[1];
    int k = a.shape()[1];

    // Create buffers
    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * k * sizeof(float), (void*)a.data(), &err);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k * n * sizeof(float), (void*)b.data(), &err);
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m * n * sizeof(float), NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    // Execute kernel
    size_t global_work_size[2] = {static_cast<size_t>(m), static_cast<size_t>(n)};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel: " << err << std::endl;
        return;
    }

    // Read result
    err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, m * n * sizeof(float), result.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error reading result buffer: " << err << std::endl;
        return;
    }

    // Clean up
    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(result_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

}  // namespace ops