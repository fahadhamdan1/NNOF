#include "optimization_pass.h"
#include "opencl_optimizations.h"
#include <OpenCL/opencl.h>
#include "optimization_pass_registrar.h" 

class OpenCLWorkGroupSizeOptimization : public OptimizationPass {
public:
    void apply(std::vector<std::shared_ptr<Tensor>>& tensors) override {
        for (auto& tensor : tensors) {
            // make tensor shape be multiple of 64
            auto shape = tensor->shape();
            if (shape[0] % 64 != 0) {
                int new_size = ((shape[0] / 64) + 1) * 64;
                auto new_tensor = std::make_shared<Tensor>(std::vector<int>{new_size});
                std::copy(tensor->data(), tensor->data() + shape[0], new_tensor->data());
                tensor = new_tensor;
            }
        }
    }
};

REGISTER_OPTIMIZATION_PASS("opencl_workgroup_size", OpenCLWorkGroupSizeOptimization);


void register_opencl_optimizations() {
}