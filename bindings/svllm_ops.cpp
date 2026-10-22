#include <torch/extension.h>

#include "ops/rms_norm.h"

PYBIND11_MODULE(svllm_ops, m) {
  m.attr("__version__") = SVLLM_VERSION;
#ifdef SVLLM_CUDA
  m.attr("built_with_cuda") = true;
#else
  m.attr("built_with_cuda") = false;
#endif
  m.def("rms_norm", &svllm::ops::rms_norm, py::arg("x"), py::arg("weight"), py::arg("eps"));
}
