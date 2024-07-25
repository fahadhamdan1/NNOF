#include "tensor.h"
#include "ops.h"
#include "scheduler.h"
#include "optimization_pass.h"
#include "optimization_pass_registrar.h"
#include <iostream>
#include <chrono>

int main() {
    // init tensors
    auto a = std::make_shared<Tensor>(std::vector<int>{1024});
    auto b = std::make_shared<Tensor>(std::vector<int>{1024});
    auto result = std::make_shared<Tensor>(std::vector<int>{1024});

    // making sample tensors with some data
    for (int i = 0; i < 1024; ++i) {
        a->data()[i] = static_cast<float>(i);
        b->data()[i] = static_cast<float>(1024 - i);
    }

    // doing optimizations
    std::vector<std::shared_ptr<Tensor>> tensors = {a, b, result};
    auto optimization = OptimizationPassRegistrar::getInstance().createPass("opencl_workgroup_size");
    if (optimization) {
        optimization->apply(tensors);
    } else {
        std::cerr << "Failed to create optimization pass" << std::endl;
    }

    // doing addition
    auto device = Scheduler::select_device(*a, *b);
    auto start = std::chrono::high_resolution_clock::now();
    if (device == Device::CPU) {
        ops::add_cpu(*a, *b, *result);
    } else {
        ops::add_gpu(*a, *b, *result);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // print results
    std::cout << "Addition performed on " << (device == Device::CPU ? "CPU" : "GPU") << std::endl;
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;
    std::cout << "First few results: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << result->data()[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}