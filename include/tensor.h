#pragma once
#include <vector>
#include <memory>

class Tensor {
public:
    Tensor(const std::vector<int>& shape, float* data = nullptr);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    ~Tensor();

    const std::vector<int>& shape() const { return shape_; }
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }

private:
    std::vector<int> shape_;
    std::unique_ptr<float[]> data_;
};

//arithmetic operations
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

//element-wise operations
Tensor elementwise_multiply(const Tensor& a, const Tensor& b);