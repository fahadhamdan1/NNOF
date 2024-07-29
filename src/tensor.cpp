#include "tensor.h"
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cstring>

Tensor::Tensor(const std::vector<int>& shape, float* data)
    : shape_(shape) {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data_ = std::make_unique<float[]>(size);
    if (data) {
        std::memcpy(data_.get(), data, size * sizeof(float));
    } else {
        std::fill(data_.get(), data_.get() + size, 0.0f);
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_) {
    int size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_ = std::make_unique<float[]>(size);
    std::memcpy(data_.get(), other.data_.get(), size * sizeof(float));
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        int size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
        data_ = std::make_unique<float[]>(size);
        std::memcpy(data_.get(), other.data_.get(), size * sizeof(float));
    }
    return *this;
}

Tensor::~Tensor() = default;

Tensor operator+(const Tensor& a, const Tensor& b) {
    assert(a.shape() == b.shape());
    Tensor result(a.shape());
    int size = std::accumulate(a.shape().begin(), a.shape().end(), 1, std::multiplies<int>());
    for (int i = 0; i < size; ++i) {
        result.data()[i] = a.data()[i] + b.data()[i];
    }
    return result;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    assert(a.shape() == b.shape());
    Tensor result(a.shape());
    int size = std::accumulate(a.shape().begin(), a.shape().end(), 1, std::multiplies<int>());
    for (int i = 0; i < size; ++i) {
        result.data()[i] = a.data()[i] - b.data()[i];
    }
    return result;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    assert(a.shape() == b.shape());
    Tensor result(a.shape());
    int size = std::accumulate(a.shape().begin(), a.shape().end(), 1, std::multiplies<int>());
    for (int i = 0; i < size; ++i) {
        result.data()[i] = a.data()[i] * b.data()[i];
    }
    return result;
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    assert(a.shape() == b.shape());
    Tensor result(a.shape());
    int size = std::accumulate(a.shape().begin(), a.shape().end(), 1, std::multiplies<int>());
    for (int i = 0; i < size; ++i) {
        result.data()[i] = a.data()[i] / b.data()[i];
    }
    return result;
}

Tensor elementwise_multiply(const Tensor& a, const Tensor& b) {
    assert(a.shape() == b.shape());
    Tensor result(a.shape());
    int size = std::accumulate(a.shape().begin(), a.shape().end(), 1, std::multiplies<int>());
    for (int i = 0; i < size; ++i) {
        result.data()[i] = a.data()[i] * b.data()[i];
    }
    return result;
}