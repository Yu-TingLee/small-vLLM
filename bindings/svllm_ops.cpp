#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "ops/rms_norm.h"
#include "tokenizer/tokenizer.h"

PYBIND11_MODULE(svllm_ops, m) {
  m.attr("__version__") = SVLLM_VERSION;
#ifdef SVLLM_CUDA
  m.attr("built_with_cuda") = true;
#else
  m.attr("built_with_cuda") = false;
#endif
  m.def("rms_norm", &svllm::ops::rms_norm, py::arg("x"), py::arg("weight"), py::arg("eps"));

  py::class_<svllm::Tokenizer>(m, "Tokenizer")
      .def(py::init<const std::filesystem::path&>(), py::arg("tokenizer_json"))
      .def("encode", &svllm::Tokenizer::encode, py::arg("text"))
      .def(
          "decode",
          [](svllm::Tokenizer& tokenizer, const std::vector<int32_t>& token_ids) {
            return tokenizer.decode(token_ids);
          },
          py::arg("token_ids"));
}
