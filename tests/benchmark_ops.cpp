#include "ops.h"
#include "tensor.h"
#include "benchmark.h"
#include <iostream>
#include <vector>
#include <memory>
#include <random>

bool verify_results(const Tensor& a, const Tensor& b) {
    auto result1 = std::make_shared<Tensor>(a.shape());
    auto result2 = std::make_shared<Tensor>(a.shape());
    
    ops::add_cpu_baseline(a, b, *result1);
    ops::add_cpu(a, b, *result2);
    
    int size = a.shape()[0] * a.shape()[1];
    for (int i = 0; i < size; ++i) {
        if (std::abs(result1->data()[i] - result2->data()[i]) > 1e-6) {
            std::cout << "Results don't match at index " << i << std::endl;
            std::cout << "Baseline result: " << result1->data()[i] << std::endl;
            std::cout << "Optimized result: " << result2->data()[i] << std::endl;
            std::cout << "Input a: " << a.data()[i] << ", Input b: " << b.data()[i] << std::endl;
            return false;
        }
    }
    return true;
}

void benchmark_add_cpu(int size) {
    auto a = std::make_shared<Tensor>(std::vector<int>{size, size});
    auto b = std::make_shared<Tensor>(std::vector<int>{size, size});
    auto result = std::make_shared<Tensor>(std::vector<int>{size, size});
    

    // init tensors
    for (int i = 0; i < size * size; ++i) {
        a->data()[i] = static_cast<float>(i);
        b->data()[i] = static_cast<float>(i);
    }

    if (!verify_results(*a, *b)) {
        std::cout << "Error: Results don't match for size " << size << "x" << size << std::endl;
        return;
    }

    const int warmup_iterations = 10;
    const int timing_iterations = 100;

    //Warm-up runs
    for (int i = 0; i < warmup_iterations; ++i) {
        ops::add_cpu_baseline(*a, *b, *result);
        ops::add_cpu(*a, *b, *result);
    }

    // Timing runs
    auto time_operation = [&](auto func) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timing_iterations; ++i) {
            func(*a, *b, *result);
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / timing_iterations;
    };

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

    double baseline_time = time_operation(ops::add_cpu_baseline);
    double optimized_time = time_operation(ops::add_cpu);

    double latency_improvement = (baseline_time - optimized_time) / baseline_time * 100.0;
    double throughput_improvement = (baseline_time / optimized_time - 1) * 100.0;

    std::cout << "Improvements for " << benchmark_name << ":" << std::endl;
    std::cout << "  Optimized Latency: " << result_optimized.latency << " ms" << std::endl;
    std::cout << "    Latency improvement: " << latency_improvement << " %" << std::endl;
    std::cout << "  Optimized Throughput: " << result_optimized.throughput << " ops/s" << std::endl;
    std::cout << "    Throughput improvement: " << throughput_improvement << " %" << std::endl;
    std::cout << "  Memory usage after optimized: " << result_optimized.memory_usage << " bytes" << std::endl;
    std::cout << std::endl;

}

bool verify_matmul_results(const Tensor& a, const Tensor& b) {
    auto result1 = std::make_shared<Tensor>(std::vector<int>{a.shape()[0], b.shape()[1]});
    auto result2 = std::make_shared<Tensor>(std::vector<int>{a.shape()[0], b.shape()[1]});
    
    ops::matmul_cpu_baseline(a, b, *result1);
    ops::matmul_cpu(a, b, *result2);
    
    int size = result1->shape()[0] * result1->shape()[1];
    for (int i = 0; i < size; ++i) {
        float rel_error = std::abs(result1->data()[i] - result2->data()[i]) / 
                          (std::abs(result1->data()[i]) + 1e-6);
        if (rel_error > 1e-5) {
            std::cout << "Matmul results don't match at index " << i << std::endl;
            std::cout << "Baseline result: " << result1->data()[i] << std::endl;
            std::cout << "Optimized result: " << result2->data()[i] << std::endl;
            std::cout << "Relative error: " << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

void benchmark_matmul_cpu(int m, int n, int k) {
    auto a = std::make_shared<Tensor>(std::vector<int>{m, k});
    auto b = std::make_shared<Tensor>(std::vector<int>{k, n});
    auto result = std::make_shared<Tensor>(std::vector<int>{m, n});

    // init tensors with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < m * k; ++i) {
        a->data()[i] = dis(gen);
    }
    for (int i = 0; i < k * n; ++i) {
        b->data()[i] = dis(gen);
    }

    if (!verify_matmul_results(*a, *b)) {
        std::cout << "Error: Matmul results don't match for size " << m << "x" << k << " * " << k << "x" << n << std::endl;
        return;
    }

    const int warmup_iterations = 5;
    const int timing_iterations = 20;

    for (int i = 0; i < warmup_iterations; ++i) {
        ops::matmul_cpu_baseline(*a, *b, *result);
        ops::matmul_cpu(*a, *b, *result);
    }

    std::vector<std::shared_ptr<Tensor>> tensors = {a, b, result};

    auto matmul_func_optimized = [](const std::vector<std::shared_ptr<Tensor>>& t) {
        ops::matmul_cpu(*t[0], *t[1], *t[2]);
    };

    auto matmul_func_baseline = [](const std::vector<std::shared_ptr<Tensor>>& t) {
        ops::matmul_cpu_baseline(*t[0], *t[1], *t[2]);
    };

    std::string benchmark_name = "CPU Matrix Multiplication " + std::to_string(m) + "x" + std::to_string(k) + " * " + std::to_string(k) + "x" + std::to_string(n);
    Benchmark::Result result_optimized = Benchmark::run(benchmark_name + " (Optimized)", matmul_func_optimized, tensors, timing_iterations);
    Benchmark::Result result_baseline = Benchmark::run(benchmark_name + " (Baseline)", matmul_func_baseline, tensors, timing_iterations);

    double latency_improvement = (result_baseline.latency - result_optimized.latency) / result_baseline.latency * 100.0;
    double throughput_improvement = (result_optimized.throughput / result_baseline.throughput - 1) * 100.0;

    std::cout << "Improvements for " << benchmark_name << ":" << std::endl;
    std::cout << "  Baseline Latency: " << result_baseline.latency << " ms" << std::endl;
    std::cout << "  Optimized Latency: " << result_optimized.latency << " ms" << std::endl;
    std::cout << "    Latency improvement: " << latency_improvement << "%" << std::endl;
    std::cout << "  Baseline Throughput: " << result_baseline.throughput << " ops/s" << std::endl;
    std::cout << "  Optimized Throughput: " << result_optimized.throughput << " ops/s" << std::endl;
    std::cout << "    Throughput improvement: " << throughput_improvement << "%" << std::endl;
    std::cout << "  Memory usage: " << result_optimized.memory_usage << " bytes" << std::endl;
    std::cout << std::endl;
}


int main() {
    std::vector<int> sizes = {128, 256, 512, 1024};

    std::cout << "Benchmarking Addition:" << std::endl;
    for (int size : sizes) {
        benchmark_add_cpu(size);
        std::cout << std::endl;
    }

    std::cout << "Benchmarking Matrix Multiplication:" << std::endl;
    for (int size : sizes) {
        benchmark_matmul_cpu(size, size, size);  //square
        std::cout << std::endl;
    }

    // non-square
    std::vector<std::tuple<int, int, int>> matmul_sizes = {
        {64, 64, 1024},    
        {512, 1024, 256},  // large
        {2048, 512, 2048}  // hella large matrices
    };

    for (const auto& [m, n, k] : matmul_sizes) {
        benchmark_matmul_cpu(m, n, k);
        std::cout << std::endl;
    }

    return 0;
}
