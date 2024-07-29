#include "activation_functions.h"
#include <cmath>

namespace activation {

Tensor relu(const Tensor& input) {
    Tensor output(input.shape());
    for (int i = 0; i < input.shape()[0] * input.shape()[1]; ++i) {
        output.data()[i] = std::max(0.0f, input.data()[i]);
    }
    return output;
}

Tensor relu_derivative(const Tensor& input) {
    Tensor output(input.shape());
    for (int i = 0; i < input.shape()[0] * input.shape()[1]; ++i) {
        output.data()[i] = input.data()[i] > 0 ? 1.0f : 0.0f;
    }
    return output;
}

Tensor sigmoid(const Tensor& input) {
    Tensor output(input.shape());
    for (int i = 0; i < input.shape()[0] * input.shape()[1]; ++i) {
        output.data()[i] = 1.0f / (1.0f + std::exp(-input.data()[i]));
    }
    return output;
}

Tensor sigmoid_derivative(const Tensor& input) {
    Tensor output(input.shape());
    for (int i = 0; i < input.shape()[0] * input.shape()[1]; ++i) {
        float s = 1.0f / (1.0f + std::exp(-input.data()[i]));
        output.data()[i] = s * (1.0f - s);
    }
    return output;
}

Tensor tanh(const Tensor& input) {
    Tensor output(input.shape());
    for (int i = 0; i < input.shape()[0] * input.shape()[1]; ++i) {
        output.data()[i] = std::tanh(input.data()[i]);
    }
    return output;
}

Tensor tanh_derivative(const Tensor& input) {
    Tensor output(input.shape());
    for (int i = 0; i < input.shape()[0] * input.shape()[1]; ++i) {
        float t = std::tanh(input.data()[i]);
        output.data()[i] = 1.0f - t * t;
    }
    return output;
}

}