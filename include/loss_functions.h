#pragma once
#include "tensor.h"

namespace loss {

float mse(const Tensor& predictions, const Tensor& targets);
Tensor mse_gradient(const Tensor& predictions, const Tensor& targets);

}