#pragma once

#include "optimization_pass.h"
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>

class OptimizationPassRegistrar {
public:
    using CreatorFunction = std::function<std::unique_ptr<OptimizationPass>()>;

    static OptimizationPassRegistrar& getInstance() {
        static OptimizationPassRegistrar instance;
        return instance;
    }

    void registerPass(const std::string& name, CreatorFunction creator) {
        registry[name] = creator;
    }

    std::unique_ptr<OptimizationPass> createPass(const std::string& name) {
        auto it = registry.find(name);
        if (it != registry.end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    OptimizationPassRegistrar() = default;
    std::unordered_map<std::string, CreatorFunction> registry;
};

#define REGISTER_OPTIMIZATION_PASS(name, classname) \
    static OptimizationPassRegistrar::CreatorFunction creator##classname = []() { \
        return std::make_unique<classname>(); \
    }; \
    static OptimizationPassRegistrar::CreatorFunction registrar##classname = \
        (OptimizationPassRegistrar::getInstance().registerPass(name, creator##classname), creator##classname);