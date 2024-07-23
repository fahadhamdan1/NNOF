#pragma once
#include "tensor.h"

enum class Device { CPU, GPU };

class Scheduler {
public:
    static Device select_device(const Tensor& a, const Tensor& b);
};