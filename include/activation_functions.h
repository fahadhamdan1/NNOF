#pragma once

#include "tensor.h"

namespace activation {

Tensor relu(const Tensor& input);
Tensor relu_derivative(const Tensor& input);

Tensor sigmoid(const Tensor& input);
Tensor sigmoid_derivative(const Tensor& input);

Tensor tanh(const Tensor& input);
Tensor tanh_derivative(const Tensor& input);

}