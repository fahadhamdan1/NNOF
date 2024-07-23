#include "benchmark.h"
#include "ops.h"

double Benchmark::measure_latency(const std::vector<std::shared_ptr<Tensor>>& tensors) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform some operations on the tensors
    for (size_t i = 1; i < tensors.size(); ++i) {
        ops::add_cpu(*tensors[i-1], *tensors[i], *tensors[i]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

double Benchmark::measure_throughput(const std::vector<std::shared_ptr<Tensor>>& tensors, int num_iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        for (size_t j = 1; j < tensors.size(); ++j) {
            ops::add_cpu(*tensors[j-1], *tensors[j], *tensors[j]);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return num_iterations / duration.count();
}

size_t Benchmark::measure_memory_usage(const std::vector<std::shared_ptr<Tensor>>& tensors) {
    size_t total_memory = 0;
    for (const auto& tensor : tensors) {
        total_memory += tensor->shape()[0] * sizeof(float);
    }
    return total_memory;
}