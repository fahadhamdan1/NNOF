#pragma once

#include "fully_connected_layer.h"
#include "convolutional_layer.h"
#include "tensor.h"
#include <vector>
#include <memory>

class Network {
public:
    void add_fully_connected_layer(int input_size, int output_size);
    void add_convolutional_layer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
    Tensor forward(const Tensor& input);
    void train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs, float learning_rate);

private:
    std::vector<std::unique_ptr<FullyConnectedLayer>> fc_layers;
    std::vector<std::unique_ptr<ConvolutionalLayer>> conv_layers;
    std::vector<Tensor> activations;
};