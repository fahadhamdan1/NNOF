cmake_minimum_required(VERSION 3.12)
project(ANNOF VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)