#include "fully_connected_layer.h"
#include "ops.h"
#include "gpu_operations.h"
#include <iostream>
#include <immintrin.h>
#include <random>
#include <stdexcept>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) {
    // init weights and bias with random values
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

Tensor FullyConnectedLayer::forward_cpu(const Tensor& input) {
    this->input = std::make_shared<Tensor>(input);
    Tensor output(std::vector<int>{input.shape()[0], weights->shape()[1]});
    
    int m = input.shape()[0];
    int n = weights->shape()[1];
    int k = weights->shape()[0];
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int l = 0; l < k; ++l) {
                __m256 a = _mm256_set1_ps(input.data()[i * k + l]);
                __m256 b = _mm256_loadu_ps(&weights->data()[l * n + j]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(&output.data()[i * n + j], sum);
        }
        
        // remaining elems
        for (int j = n - n % 8; j < n; ++j) {
            float sum = 0;
            for (int l = 0; l < k; ++l) {
                sum += input.data()[i * k + l] * weights->data()[l * n + j];
            }
            output.data()[i * n + j] = sum;
        }
    }
    
    // bias
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; j += 8) {
            __m256 out = _mm256_loadu_ps(&output.data()[i * n + j]);
            __m256 b = _mm256_loadu_ps(&bias->data()[j]);
            _mm256_storeu_ps(&output.data()[i * n + j], _mm256_add_ps(out, b));
        }
        
        for (int j = n - n % 8; j < n; ++j) {
            output.data()[i * n + j] += bias->data()[j];
        }
    }
    
    return output;
}

Tensor FullyConnectedLayer::forward_gpu(const Tensor& input) {
    try {
        return gpu_operations::fully_connected_forward(input, *weights, *bias);
    } catch (const std::exception& e) {
        std::cerr << "GPU forward pass failed: " << e.what() << std::endl;
        std::cerr << "Falling back to CPU implementation." << std::endl;
        return forward_cpu(input);
    }
}

Tensor FullyConnectedLayer::forward(const Tensor& input, bool use_gpu) {
    if (use_gpu) {
        return forward_gpu(input);
    } else {
        return forward_cpu(input);
    }
}

Tensor FullyConnectedLayer::backward(const Tensor& output_gradient, float learning_rate) {
    int batch_size = output_gradient.shape()[0];
    int input_size = weights->shape()[0];
    int output_size = weights->shape()[1];

    Tensor input_gradient({batch_size, input_size});
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_size; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < output_size; ++k) {
                __m256 grad = _mm256_set1_ps(output_gradient.data()[i * output_size + k]);
                __m256 w = _mm256_loadu_ps(&weights->data()[j * output_size + k]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(grad, w));
            }
            _mm256_storeu_ps(&input_gradient.data()[i * input_size + j], sum);
        }
        
        for (int j = input_size - input_size % 8; j < input_size; ++j) {
            float sum = 0;
            for (int k = 0; k < output_size; ++k) {
                sum += output_gradient.data()[i * output_size + k] * weights->data()[j * output_size + k];
            }
            input_gradient.data()[i * input_size + j] = sum;
        }
    }

    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; j += 8) {
            __m256 w_update = _mm256_setzero_ps();
            for (int k = 0; k < batch_size; ++k) {
                __m256 grad = _mm256_loadu_ps(&output_gradient.data()[k * output_size + j]);
                __m256 in = _mm256_set1_ps(input->data()[k * input_size + i]);
                w_update = _mm256_add_ps(w_update, _mm256_mul_ps(grad, in));
            }
            __m256 w = _mm256_loadu_ps(&weights->data()[i * output_size + j]);
            w = _mm256_sub_ps(w, _mm256_mul_ps(_mm256_set1_ps(learning_rate), w_update));
            _mm256_storeu_ps(&weights->data()[i * output_size + j], w);
        }
        
        for (int j = output_size - output_size % 8; j < output_size; ++j) {
            float w_update = 0;
            for (int k = 0; k < batch_size; ++k) {
                w_update += output_gradient.data()[k * output_size + j] * input->data()[k * input_size + i];
            }
            weights->data()[i * output_size + j] -= learning_rate * w_update;
        }
    }

    for (int j = 0; j < output_size; j += 8) {
        __m256 b_update = _mm256_setzero_ps();
        for (int i = 0; i < batch_size; ++i) {
            __m256 grad = _mm256_loadu_ps(&output_gradient.data()[i * output_size + j]);
            b_update = _mm256_add_ps(b_update, grad);
        }
        __m256 b = _mm256_loadu_ps(&bias->data()[j]);
        b = _mm256_sub_ps(b, _mm256_mul_ps(_mm256_set1_ps(learning_rate), b_update));
        _mm256_storeu_ps(&bias->data()[j], b);
    }
    
    for (int j = output_size - output_size % 8; j < output_size; ++j) {
        float b_update = 0;
        for (int i = 0; i < batch_size; ++i) {
            b_update += output_gradient.data()[i * output_size + j];
        }
        bias->data()[j] -= learning_rate * b_update;
    }

    return input_gradient;
}