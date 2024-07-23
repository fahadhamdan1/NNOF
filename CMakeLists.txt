cmake_minimum_required(VERSION 3.12)
project(ANNOF VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# src files
add_library(annof
    src/tensor.cpp
    src/ops_cpu.cpp
    src/ops_opencl.cpp
    src/scheduler.cpp
    src/optimization_pass.cpp
    src/benchmark.cpp
)

# Link against OpenCL
target_link_libraries(annof ${OpenCL_LIBRARIES})

# testssss
add_subdirectory(tests)
add_subdirectory(examples)


