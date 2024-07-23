#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(annof_core, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&>())
        .def("shape", &Tensor::shape);

    m.def("add_cpu", &ops::add_cpu, "Perform addition on CPU");
    m.def("add_gpu", &ops::add_gpu, "Perform addition on GPU using OpenCL");
}