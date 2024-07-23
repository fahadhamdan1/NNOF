#pragma once
#include "tensor.h"
#include <chrono>
#include <vector>

class Benchmark {
public:
    static double measure_latency(const std::vector<std::shared_ptr<Tensor>>& tensors);
    static double measure_throughput(const std::vector<std::shared_ptr<Tensor>>& tensors, int num_iterations);
    static size_t measure_memory_usage(const std::vector<std::shared_ptr<Tensor>>& tensors);
};