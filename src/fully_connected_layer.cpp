#include "fully_connected_layer.h"
#include "ops.h"
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) {
    // Initialize weights and bias with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    weights = std::make_shared<Tensor>(std::vector<int>{input_size, output_size});
    bias = std::make_shared<Tensor>(std::vector<int>{1, output_size});

    for (int i = 0; i < input_size * output_size; ++i) {
        weights->data()[i] = d(gen) / std::sqrt(input_size);
    }

    for (int i = 0; i < output_size; ++i) {
        bias->data()[i] = 0;
    }
}

Tensor FullyConnectedLayer::forward(const Tensor& input) {
    this->input = std::make_shared<Tensor>(input);
    Tensor output(std::vector<int>{input.shape()[0], weights->shape()[1]});
    ops::matmul_cpu(*this->input, *weights, output);
    
    // Add bias
    for (int i = 0; i < output.shape()[0]; ++i) {
        for (int j = 0; j < output.shape()[1]; ++j) {
            output.data()[i * output.shape()[1] + j] += bias->data()[j];
        }
    }

    return output;
}

Tensor FullyConnectedLayer::backward(const Tensor& output_gradient) {
    // Compute gradients
    Tensor weights_gradient(weights->shape());
    ops::matmul_cpu(*input, output_gradient, weights_gradient);

    Tensor input_gradient(input->shape());
    Tensor weights_transposed(std::vector<int>{weights->shape()[1], weights->shape()[0]});
    // Transpose weights
    for (int i = 0; i < weights->shape()[0]; ++i) {
        for (int j = 0; j < weights->shape()[1]; ++j) {
            weights_transposed.data()[j * weights->shape()[0] + i] = weights->data()[i * weights->shape()[1] + j];
        }
    }
    ops::matmul_cpu(output_gradient, weights_transposed, input_gradient);

    // Update weights and bias
    for (int i = 0; i < weights->shape()[0] * weights->shape()[1]; ++i) {
        weights->data()[i] -= learning_rate * weights_gradient.data()[i];
    }

    for (int i = 0; i < bias->shape()[1]; ++i) {
        float bias_gradient = 0;
        for (int j = 0; j < output_gradient.shape()[0]; ++j) {
            bias_gradient += output_gradient.data()[j * output_gradient.shape()[1] + i];
        }
        bias->data()[i] -= learning_rate * bias_gradient;
    }

    return input_gradient;
}

void FullyConnectedLayer::update_parameters(float learning_rate) {
    // This method is left empty because we're updating parameters in the backward pass
    // You might want to implement more sophisticated optimizers here in the future
}