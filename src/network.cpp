#include "network.h"
#include "activation_functions.h"
#include "loss_functions.h"
#include <cmath>
#include <iostream>

void Network::add_layer(std::unique_ptr<FullyConnectedLayer> layer) {
    layers.push_back(std::move(layer));
}

Tensor Network::forward(const Tensor& input) {
    activations.clear();
    activations.push_back(input);

    Tensor current = input;
    for (const auto& layer : layers) {
        current = layer->forward(current);
        current = activation::relu(current);  //using relu for now
        activations.push_back(current);
    }

    return current;
}

void Network::backward(const Tensor& target, float learning_rate) {
    Tensor error = activations.back() - target;

    // backpropagataion
    for (int i = layers.size() - 1; i >= 0; --i) {
        Tensor activation_gradient = activation::relu_derivative(activations[i + 1]);
        error = error * activation_gradient;
        error = layers[i]->backward(error, learning_rate);
    }
}

void Network::train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor predictions = forward(inputs[i]);
            total_loss += loss::mse(predictions, targets[i]);
            
            Tensor error = loss::mse_gradient(predictions, targets[i]);
            backward(error, learning_rate);
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", Loss: " << total_loss / inputs.size() << std::endl;
    }
}