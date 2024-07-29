#pragma once

#include "tensor.h"
#include <memory>

class FullyConnectedLayer {
public:
    FullyConnectedLayer(int input_size, int output_size);
    
    Tensor forward(const Tensor& input, bool use_gpu = false);
    Tensor forward_cpu(const Tensor& input);
    Tensor forward_gpu(const Tensor& input);
    Tensor backward(const Tensor& output_gradient, float learning_rate);
    

private:
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> input;
};