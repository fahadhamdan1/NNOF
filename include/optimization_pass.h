#pragma once
#include "tensor.h"
#include <vector>
#include <memory>

class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;
    virtual void apply(std::vector<std::shared_ptr<Tensor>>& tensors) = 0;
};

class MemoryReductionPass : public OptimizationPass {
public:
    void apply(std::vector<std::shared_ptr<Tensor>>& tensors) override;
};

class LatencyReductionPass : public OptimizationPass {
public:
    void apply(std::vector<std::shared_ptr<Tensor>>& tensors) override;
};

class OpenCLOptimizationPass : public OptimizationPass {
public:
    void apply(std::vector<std::shared_ptr<Tensor>>& tensors) override;
};  