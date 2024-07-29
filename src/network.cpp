#include "network.h"
#include "activation_functions.h"
#include "loss_functions.h"
#include <iostream>

void Network::add_fully_connected_layer(int input_size, int output_size) {
    fc_layers.push_back(std::make_unique<FullyConnectedLayer>(input_size, output_size));
}

void Network::add_convolutional_layer(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    conv_layers.push_back(std::make_unique<ConvolutionalLayer>(in_channels, out_channels, kernel_size, stride, padding));
}

Tensor Network::forward(const Tensor& input) {
    Tensor current = input;
    
    for (const auto& layer : conv_layers) {
        current = layer->forward(current);
        current = activation::relu(current);
    }
    
    // Flatten the output of convolutional layers for fully connected layers
    if (!fc_layers.empty() && current.shape().size() > 2) {
        int flat_size = current.shape()[1] * current.shape()[2] * current.shape()[3];
        current = Tensor({current.shape()[0], flat_size}, current.data());
    }
    
    for (const auto& layer : fc_layers) {
        current = layer->forward(current);
        current = activation::relu(current);
    }
    
    return current;
}

void Network::train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            //forward pass and compute loss
            Tensor predictions = forward(inputs[i]);
            total_loss += loss::mse(predictions, targets[i]);
            
            //assumes only fully connected layers for now, no real backward pass
            Tensor error = loss::mse_gradient(predictions, targets[i]);
            
            for (int j = fc_layers.size() - 1; j >= 0; --j) {
                error = fc_layers[j]->backward(error, learning_rate);
            }
            
        }
         
        float avg_loss = total_loss / inputs.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", Average Loss: " << avg_loss << std::endl;
    }
}