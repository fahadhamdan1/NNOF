#pragma once
#include "tensor.h"

namespace ops {


void add_cpu_baseline(const Tensor& a, const Tensor& b, Tensor& result);
void add_cpu(const Tensor& a, const Tensor& b, Tensor& result);
void add_gpu(const Tensor& a, const Tensor& b, Tensor& result);

void matmul_cpu_baseline(const Tensor& a, const Tensor& b, Tensor& result);
void matmul_cpu(const Tensor& a, const Tensor& b, Tensor& result);
void matmul_gpu(const Tensor& a, const Tensor& b, Tensor& result);

}