#include "benchmark.h"
#include "fully_connected_layer.h"
#include "gpu_operations.h"
#include <iostream>
#include <vector>
#include <random>

void benchmark_fully_connected_layer(int input_size, int output_size, int batch_size) {
    FullyConnectedLayer layer(input_size, output_size);
    
    auto input = std::make_shared<Tensor>(std::vector<int>{batch_size, input_size});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < batch_size * input_size; ++i) {
        input->data()[i] = dis(gen);
    }

    std::vector<std::shared_ptr<Tensor>> tensors = {input};

    // CPU forward pass
    auto cpu_forward = [&](const std::vector<std::shared_ptr<Tensor>>& t) {
        layer.forward(*t[0], false);
    };
    auto cpu_result = Benchmark::run("CPU Forward", cpu_forward, tensors);
    
    // GPU forward pass
    auto gpu_forward = [&](const std::vector<std::shared_ptr<Tensor>>& t) {
        layer.forward(*t[0], true);
    };
    auto gpu_result = Benchmark::run("GPU Forward", gpu_forward, tensors);


    std::cout << "Fully Connected Layer Benchmark:" << std::endl;
    std::cout << "Input size: " << input_size << ", Output size: " << output_size << ", Batch size: " << batch_size << std::endl;
    Benchmark::printResults("CPU Forward Pass", cpu_result);
    Benchmark::printResults("GPU Forward Pass", gpu_result);
    std::cout << "GPU Speedup: " << cpu_result.latency / gpu_result.latency << "x" << std::endl;
    std::cout << std::endl;
}

int main() {
    gpu_operations::initialize();

    std::vector<int> input_sizes = {128, 256, 512, 1024};
    std::vector<int> output_sizes = {64, 128, 256, 512};
    std::vector<int> batch_sizes = {32, 64, 128, 256};

    for (int input_size : input_sizes) {
        for (int output_size : output_sizes) {
            for (int batch_size : batch_sizes) {
                benchmark_fully_connected_layer(input_size, output_size, batch_size);
            }
        }
    }

    gpu_operations::cleanup();

    return 0;
}
