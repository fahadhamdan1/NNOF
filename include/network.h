#pragma once

#include <vector>
#include <memory>
#include "tensor.h"
#include "fully_connected_layer.h"

class Network {
public:
    void add_layer(std::unique_ptr<FullyConnectedLayer> layer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& target, float learning_rate);
    void train(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int epochs, float learning_rate);

private:
    std::vector<std::unique_ptr<FullyConnectedLayer>> layers;
    std::vector<Tensor> activations;
};