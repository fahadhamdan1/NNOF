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

    auto add_func = [](const std::vector<std::shared_ptr<Tensor>>& t) {
        ops::add_cpu(*t[0], *t[1], *t[2]);
    };

    std::string benchmark_name = "CPU Addition " + std::to_string(size) + "x" + std::to_string(size);
    Benchmark::Result benchmark_result = Benchmark::run(benchmark_name, add_func, tensors);

    // You can use the result for further analysis if needed
    std::cout << "Latency: " << benchmark_result.latency << " ms" << std::endl;
    std::cout << "Throughput: " << benchmark_result.throughput << " ops/s" << std::endl;
    std::cout << "Memory usage: " << benchmark_result.memory_usage << " bytes" << std::endl;
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