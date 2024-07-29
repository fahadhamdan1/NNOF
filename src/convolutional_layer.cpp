#include "convolutional_layer.h"
#include <random>
#include <cmath>

ConvolutionalLayer::ConvolutionalLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    
    // Initialize weights and bias
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, std::sqrt(2.0 / (in_channels * kernel_size * kernel_size)));

    weights_ = std::make_shared<Tensor>(std::vector<int>{out_channels_, in_channels_, kernel_size_, kernel_size_});
    bias_ = std::make_shared<Tensor>(std::vector<int>{out_channels_});

    int weight_size = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;
    for (int i = 0; i < weight_size; ++i) {
        weights_->data()[i] = d(gen);
    }

    for (int i = 0; i < out_channels_; ++i) {
        bias_->data()[i] = 0;
    }
}

Tensor ConvolutionalLayer::forward(const Tensor& input) {
    input_ = std::make_shared<Tensor>(input);
    Tensor padded_input = pad_input(input);
    
    int batch_size = input.shape()[0];
    int input_height = input.shape()[2];
    int input_width = input.shape()[3];
    int output_height = (input_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_width = (input_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    Tensor output(std::vector<int>{batch_size, out_channels_, output_height, output_width});

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    float sum = bias_->data()[oc];
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ + kh;
                                int iw = ow * stride_ + kw;
                                sum += padded_input.data()[b * in_channels_ * (input_height + 2 * padding_) * (input_width + 2 * padding_) +
                                                         ic * (input_height + 2 * padding_) * (input_width + 2 * padding_) +
                                                         ih * (input_width + 2 * padding_) + iw] *
                                       weights_->data()[oc * in_channels_ * kernel_size_ * kernel_size_ +
                                                        ic * kernel_size_ * kernel_size_ +
                                                        kh * kernel_size_ + kw];
                            }
                        }
                    }
                    output.data()[b * out_channels_ * output_height * output_width +
                                  oc * output_height * output_width +
                                  oh * output_width + ow] = sum;
                }
            }
        }
    }

    return output;
}


Tensor ConvolutionalLayer::pad_input(const Tensor& input) const {
    if (padding_ == 0) {
        return input;
    }

    int batch_size = input.shape()[0];
    int channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];

    Tensor padded(std::vector<int>{batch_size, channels, height + 2 * padding_, width + 2 * padding_});

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    padded.data()[b * channels * (height + 2 * padding_) * (width + 2 * padding_) +
                                  c * (height + 2 * padding_) * (width + 2 * padding_) +
                                  (h + padding_) * (width + 2 * padding_) + (w + padding_)] =
                        input.data()[b * channels * height * width +
                                     c * height * width +
                                     h * width + w];
                }
            }
        }
    }

    return padded;
}