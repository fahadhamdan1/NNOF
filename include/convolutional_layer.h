#pragma once

#include "tensor.h"
#include <vector>
#include <memory>

class ConvolutionalLayer {
public:
    ConvolutionalLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
    
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& output_gradient, float learning_rate);
    
    void update_parameters(float learning_rate);

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    
    std::shared_ptr<Tensor> weights_;
    std::shared_ptr<Tensor> bias_;
    std::shared_ptr<Tensor> input_;
    
    Tensor pad_input(const Tensor& input) const;
    Tensor convolve(const Tensor& input, const Tensor& kernel) const;
};