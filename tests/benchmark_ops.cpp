#include "ops.h"
#include "tensor.h"
#include "benchmark.h"
#include <iostream>
#include <vector>
#include <memory>

void benchmark_add_cpu(int size) {
    auto a = std::make_shared<Tensor>(std::vector<int>{size, size});
    auto b = std::make_shared<Tensor>(std::vector<int>{size, size});
    auto result = std::make_shared<Tensor>(std::vector<int>{size, size});

    // init tensors
    for (int i = 0; i < size * size; ++i) {
        a->data()[i] = static_cast<float>(i);
        b->data()[i] = static_cast<float>(i);
    }

    std::vector<std::shared_ptr<Tensor>> tensors = {a, b, result};

    auto add_func_optimized = [](const std::vector<std::shared_ptr<Tensor>>& t) {
        ops::add_cpu(*t[0], *t[1], *t[2]);
    };

    auto add_func_baseline = [](const std::vector<std::shared_ptr<Tensor>>& t) {
        ops::add_cpu_baseline(*t[0], *t[1], *t[2]);
    };

    std::string benchmark_name = "CPU Addition " + std::to_string(size) + "x" + std::to_string(size);
    Benchmark::Result result_optimized = Benchmark::run(benchmark_name + " (Optimized)", add_func_optimized, tensors);
    Benchmark::Result result_baseline = Benchmark::run(benchmark_name + " (Baseline)", add_func_baseline, tensors);


    // You can use the result for further analysis if needed

    double latency_improvement = (result_baseline.latency - result_optimized.latency) / result_baseline.latency * 100.0;
    double throughput_improvement = (result_optimized.throughput - result_baseline.throughput) / result_baseline.throughput * 100.0;

    std::cout << "Improvements for " << benchmark_name << ":" << std::endl;
    std::cout << "  Optimized Latency: " << result_optimized.latency << " ms" << std::endl;
    std::cout << "    Latency improvement: " << latency_improvement << "%" << std::endl;
    std::cout << "  Optimized Throughput: " << result_optimized.throughput << " ops/s" << std::endl;
    std::cout << "    Throughput improvement: " << throughput_improvement << "%" << std::endl;
    std::cout << "  Optimized Memory usage: " << result_optimized.memory_usage << " bytes" << std::endl;
    std::cout << std::endl;

}

// If you have GPU (OpenCL) implementation, you can add a similar function:
/*
void benchmark_add_gpu(int size) {
    // Similar to benchmark_add_cpu, but use ops::add_gpu
    // You'll need to handle OpenCL context creation and memory transfers
}
*/

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024};

    for (int size : sizes) {
        benchmark_add_cpu(size);
        std::cout << std::endl;
    }

    // If you implement GPU benchmarks:
    /*
    for (int size : sizes) {
        benchmark_add_gpu(size);
        std::cout << std::endl;
    }
    */

    return 0;
}