#include "tensor.h"
#include <numeric>
#include <algorithm>

Tensor::Tensor(const std::vector<int>& shape, float* data)
    : shape_(shape) {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data_ = std::make_unique<float[]>(size);
    if (data) {
        std::copy(data, data + size, data_.get());
    }
}

Tensor::~Tensor() = default;