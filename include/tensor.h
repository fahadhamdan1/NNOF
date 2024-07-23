#pragma once
#include <vector>
#include <memory>

class Tensor {
public:
    Tensor(const std::vector<int>& shape, float* data = nullptr);
    ~Tensor();

    const std::vector<int>& shape() const { return shape_; }
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }

private:
    std::vector<int> shape_;
    std::unique_ptr<float[]> data_;
};