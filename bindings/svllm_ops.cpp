#include <torch/extension.h>

PYBIND11_MODULE(svllm_ops, m) { m.attr("__version__") = SVLLM_VERSION; }
