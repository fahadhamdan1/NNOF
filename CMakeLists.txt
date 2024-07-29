cmake_minimum_required(VERSION 3.12)
project(ANNOF VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# src files
add_library(annof
    src/activation_functions.cpp
    src/benchmark.cpp
    src/convolutional_layer.cpp
    src/fully_connected_layer.cpp
    src/gpu_operations.cpp
    src/loss_functions.cpp
    src/network.cpp
    src/opencl_optimizations.cpp
    src/ops_cpu.cpp
    src/ops_opencl.cpp
    src/optimization_pass.cpp
    src/scheduler.cpp
    src/tensor.cpp
    include/tensor.h
    include/fully_connected_layer.h
    include/gpu_operations.h
)

target_include_directories(annof PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(annof ${OpenCL_LIBRARIES})

# demo
add_executable(demo_app examples/demo_app.cpp)
target_link_libraries(demo_app annof ${OpenCL_LIBRARIES})

# tests
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops annof)

# Benchmarks
add_executable(benchmark_ops tests/benchmark_ops.cpp)
target_link_libraries(benchmark_ops annof)

add_executable(benchmark_nn tests/benchmark_nn.cpp)
target_link_libraries(benchmark_nn annof ${OpenCL_LIBRARIES})

if(APPLE)
    target_link_libraries(benchmark_ops 
        "-framework CoreFoundation"
        "-framework OpenCL"
    )
endif()

# AVX
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no AVX support, using fallback.")
endif()