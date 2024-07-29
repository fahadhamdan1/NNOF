#pragma once

#include "tensor.h"
#include <tuple>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace gpu_operations {

void initialize();
void cleanup();

Tensor fully_connected_forward(const Tensor& input, const Tensor& weights, const Tensor& bias);
std::tuple<Tensor, Tensor, Tensor> fully_connected_backward(const Tensor& output_gradient, const Tensor& input, const Tensor& weights);

}