#include "scheduler.h"
#include <OpenCL/opencl.h>

Device Scheduler::select_device(const Tensor& a, const Tensor& b) {
    int size = a.shape()[0];
    
    // check if OpenCL GPU is available
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) return Device::CPU;
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) return Device::CPU;
    
    // I guess metric is use GPU for larger tensors
    return (size > 1000000) ? Device::GPU : Device::CPU;
}