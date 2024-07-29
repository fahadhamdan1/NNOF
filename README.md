# NNOF: Neural Network Optimization Framework

This is a C++ project aimed at implementing and optimizing neural network operations for both CPU and GPU. The primary goal is to compare performance between basic CPU operations and optimized CPU operations (SIMD), as well as to compare performance between CPU and GPU implementations of basic neural network components.

## Project Overview

This project implements:

1. A Tensor class for handling multi-dimensional data
2. CPU implementations of basic operations (addition, matrix multiplication)
3. SIMD-optimized versions of these operations using AVX instructions
4. A FullyConnectedLayer class with both CPU and GPU forward pass, using OpenCL for GPU
5. A benchmarking system to compare baseline and optimized CPU performance, and to compare CPU and GPU performance (Latency, Throughput, Memory Usage)

## Key Components

- `tensor.h/cpp`: Defines the Tensor class for data representation
- `ops_cpu.cpp`: CPU implementations of neural network operations
- `ops_opencl.cpp`: GPU (OpenCL) implementations of neural network operations
- `fully_connected_layer.h/cpp`: Implementation of a fully connected neural network layer
- `gpu_operations.h/cpp`: Wrapper for OpenCL operations
- `benchmark.h/cpp`: Benchmarking utilities

## Example Benchmarking

<img width="365" alt="image" src="https://github.com/user-attachments/assets/cf55ade3-527b-4ef0-a7b9-ec15f60696d2">

<img width="443" alt="image" src="https://github.com/user-attachments/assets/4b66aeec-13e5-471c-9c5a-2e4b40d0fd60">
