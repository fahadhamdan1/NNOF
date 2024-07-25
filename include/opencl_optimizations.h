#pragma once

#include "optimization_pass.h"
#include <vector>
#include <memory>

class OpenCLWorkGroupSizeOptimization : public OptimizationPass {
public:
    void apply(std::vector<std::shared_ptr<Tensor>>& tensors) override;
};

// declaring registration function
void register_opencl_optimizations();