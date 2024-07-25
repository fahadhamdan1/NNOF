#include "benchmark.h"
#include "ops.h"
#include <iostream>

void Benchmark::printResults(const std::string& name, const Benchmark::Result& result) {
    std::cout << "Benchmark: " << name << std::endl;
    std::cout << "  Latency: " << result.latency << " ms" << std::endl;
    std::cout << "  Throughput: " << result.throughput << " ops/s" << std::endl;
    std::cout << "  Memory usage: " << result.memory_usage << " bytes" << std::endl;
}

double Benchmark::measure_latency(std::function<void(const std::vector<std::shared_ptr<Tensor>>&)> func, const std::vector<std::shared_ptr<Tensor>>& tensors) {
    auto start = std::chrono::high_resolution_clock::now();
    func(tensors);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

double Benchmark::measure_throughput(std::function<void(const std::vector<std::shared_ptr<Tensor>>&)> func, const std::vector<std::shared_ptr<Tensor>>& tensors, int num_iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        func(tensors);
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