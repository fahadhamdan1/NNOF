#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

#include <tensor.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// for MacOS memory
#include <mach/mach.h>

class Benchmark {
public:
    struct Result {
        double latency;
        double throughput;
        size_t memory_usage;
    };

    static Result run(const std::string& name, std::function<void(const std::vector<std::shared_ptr<Tensor>>&)> func, const std::vector<std::shared_ptr<Tensor>>& tensors, int iterations = 100) {
        Result result;
        result.latency = measure_latency(func, tensors);
        result.throughput = measure_throughput(func, tensors, iterations);
        result.memory_usage = measure_memory_usage(tensors);

        return result;
    }

    static double measure_latency(std::function<void(const std::vector<std::shared_ptr<Tensor>>&)> func, const std::vector<std::shared_ptr<Tensor>>& tensors);
    static double measure_throughput(std::function<void(const std::vector<std::shared_ptr<Tensor>>&)> func, const std::vector<std::shared_ptr<Tensor>>& tensors, int num_iterations);
    static size_t measure_memory_usage(const std::vector<std::shared_ptr<Tensor>>& tensors);
    static void printResults(const std::string& name, const Result& result);
};