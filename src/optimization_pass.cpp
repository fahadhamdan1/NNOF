#include "optimization_pass.h"
#include <algorithm>

void MemoryReductionPass::apply(std::vector<std::shared_ptr<Tensor>>& tensors) {
    // Implement memory reduction 
    // simple tensor func
    if (tensors.size() < 2) return;

    auto it = std::adjacent_find(tensors.begin(), tensors.end(),
        [](const auto& a, const auto& b) {
            return a->shape() == b->shape();
        });

    if (it != tensors.end()) {
        auto next = std::next(it);
        // Fuse tensors by element-wise addition (just an example)
        for (size_t i = 0; i < (*it)->shape()[0]; ++i) {
            (*it)->data()[i] += (*next)->data()[i];
        }
        tensors.erase(next);
    }
}

void LatencyReductionPass::apply(std::vector<std::shared_ptr<Tensor>>& tensors) {
    // latency reduction techniques
    // simple operation reorderingg
    std::sort(tensors.begin(), tensors.end(),
        [](const auto& a, const auto& b) {
            return a->shape()[0] < b->shape()[0];
        });
}


#include "optimization_pass.h"
#include <OpenCL/opencl.h>

void OpenCLOptimizationPass::apply(std::vector<std::shared_ptr<Tensor>>& tensors) {
    // OpenCL-specific optimizations
    // maybe reorganize data for better memory coalescing? or adjust tensor sizes to match optimal work group sizes
    
    // placeholder for now
    for (auto& tensor : tensors) {
        // Adjust tensor shape if not a multiple of 64
        auto shape = tensor->shape();
        if (shape[0] % 64 != 0) {
            int new_size = ((shape[0] / 64) + 1) * 64;
            auto new_tensor = std::make_shared<Tensor>(std::vector<int>{new_size});
            std::copy(tensor->data(), tensor->data() + shape[0], new_tensor->data());
            tensor = new_tensor;
        }
    }
}