#include "gpu_operations.h"
#include <stdexcept>
#include <vector>
#include <iostream>

namespace gpu_operations {

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel fc_forward_kernel;
cl_kernel fc_backward_kernel;

const char* kernel_source = R"(
__kernel void fully_connected_forward(
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int input_size,
    const int output_size
) {
    int gid = get_global_id(0);
    int batch_id = gid / output_size;
    int output_id = gid % output_size;

    float sum = bias[output_id];
    for (int i = 0; i < input_size; ++i) {
        sum += input[batch_id * input_size + i] * weights[i * output_size + output_id];
    }
    output[gid] = sum;
}

__kernel void fully_connected_backward(
    __global const float* output_gradient,
    __global const float* input,
    __global const float* weights,
    __global float* input_gradient,
    __global float* weight_gradient,
    __global float* bias_gradient,
    const int input_size,
    const int output_size,
    const int batch_size
) {
    int gid = get_global_id(0);

    if (gid < input_size * output_size) {
        int input_id = gid / output_size;
        int output_id = gid % output_size;
        float grad = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad += output_gradient[b * output_size + output_id] * input[b * input_size + input_id];
        }
        weight_gradient[gid] = grad;

        if (input_id == 0) {
            float bias_grad = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                bias_grad += output_gradient[b * output_size + output_id];
            }
            bias_gradient[output_id] = bias_grad;
        }
    }

    if (gid < batch_size * input_size) {
        int batch_id = gid / input_size;
        int input_id = gid % input_size;
        float grad = 0.0f;
        for (int o = 0; o < output_size; ++o) {
            grad += output_gradient[batch_id * output_size + o] * weights[input_id * output_size + o];
        }
        input_gradient[gid] = grad;
    }
}
)";

void initialize() {
    cl_int err;

    // Get platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to get platform ID");

    // Get device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to get device ID");

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create context");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");

    // Create program
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "OpenCL program build log: " << log.data() << std::endl;
        throw std::runtime_error("Failed to build program");
    }

    // Create kernels
    fc_forward_kernel = clCreateKernel(program, "fully_connected_forward", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create forward kernel");

    fc_backward_kernel = clCreateKernel(program, "fully_connected_backward", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create backward kernel");
}

void cleanup() {
    clReleaseKernel(fc_forward_kernel);
    clReleaseKernel(fc_backward_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

Tensor fully_connected_forward(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    cl_int err;

    int batch_size = input.shape()[0];
    int input_size = input.shape()[1];
    int output_size = weights.shape()[1];

    Tensor output({batch_size, output_size});

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                         batch_size * input_size * sizeof(float), (void*)input.data(), &err);
    cl_mem weights_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                           input_size * output_size * sizeof(float), (void*)weights.data(), &err);
    cl_mem bias_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                        output_size * sizeof(float), (void*)bias.data(), &err);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                          batch_size * output_size * sizeof(float), NULL, &err);

    clSetKernelArg(fc_forward_kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(fc_forward_kernel, 1, sizeof(cl_mem), &weights_buffer);
    clSetKernelArg(fc_forward_kernel, 2, sizeof(cl_mem), &bias_buffer);
    clSetKernelArg(fc_forward_kernel, 3, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(fc_forward_kernel, 4, sizeof(int), &input_size);
    clSetKernelArg(fc_forward_kernel, 5, sizeof(int), &output_size);

    size_t global_work_size = batch_size * output_size;
    err = clEnqueueNDRangeKernel(queue, fc_forward_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue kernel");

    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, 
                              batch_size * output_size * sizeof(float), output.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to read output buffer");

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(weights_buffer);
    clReleaseMemObject(bias_buffer);
    clReleaseMemObject(output_buffer);

    return output;
}

std::tuple<Tensor, Tensor, Tensor> fully_connected_backward(const Tensor& output_gradient, const Tensor& input, const Tensor& weights) {
    cl_int err;

    int batch_size = input.shape()[0];
    int input_size = input.shape()[1];
    int output_size = weights.shape()[1];

    Tensor input_gradient({batch_size, input_size});
    Tensor weight_gradient({input_size, output_size});
    Tensor bias_gradient({1, output_size});

    cl_mem output_gradient_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                                   batch_size * output_size * sizeof(float), (void*)output_gradient.data(), &err);
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                         batch_size * input_size * sizeof(float), (void*)input.data(), &err);
    cl_mem weights_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                           input_size * output_size * sizeof(float), (void*)weights.data(), &err);
    cl_mem input_gradient_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                                  batch_size * input_size * sizeof(float), NULL, &err);
    cl_mem weight_gradient_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                                   input_size * output_size * sizeof(float), NULL, &err);
    cl_mem bias_gradient_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                                 output_size * sizeof(float), NULL, &err);

    clSetKernelArg(fc_backward_kernel, 0, sizeof(cl_mem), &output_gradient_buffer);
    clSetKernelArg(fc_backward_kernel, 1, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(fc_backward_kernel, 2, sizeof(cl_mem), &weights_buffer);
    clSetKernelArg(fc_backward_kernel, 3, sizeof(cl_mem), &input_gradient_buffer);
    clSetKernelArg(fc_backward_kernel, 4, sizeof(cl_mem), &weight_gradient_buffer);
    clSetKernelArg(fc_backward_kernel, 5, sizeof(cl_mem), &bias_gradient_buffer);
    clSetKernelArg(fc_backward_kernel, 6, sizeof(int), &input_size);
    clSetKernelArg(fc_backward_kernel, 7, sizeof(int), &output_size);
    clSetKernelArg(fc_backward_kernel, 8, sizeof(int), &batch_size);

    size_t global_work_size = std::max(input_size * output_size, batch_size * input_size);
    err = clEnqueueNDRangeKernel(queue, fc_backward_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue kernel");

    err = clEnqueueReadBuffer(queue, input_gradient_buffer, CL_TRUE, 0, 
                              batch_size * input_size * sizeof(float), input_gradient.data(), 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, weight_gradient_buffer, CL_TRUE, 0, 
                               input_size * output_size * sizeof(float), weight_gradient.data(), 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, bias_gradient_buffer, CL_TRUE, 0, 
                               output_size * sizeof(float), bias_gradient.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to read output buffers");

    clReleaseMemObject(output_gradient_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(weights_buffer);
    clReleaseMemObject(input_gradient_buffer);
    clReleaseMemObject(weight_gradient_buffer);
    clReleaseMemObject(bias_gradient_buffer);

    return std::make_tuple(input_gradient, weight_gradient, bias_gradient);
}

}