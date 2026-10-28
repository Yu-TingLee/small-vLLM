#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "tokenizer/tokenizer.h"

PYBIND11_MODULE(svllm_ops, m) {
  m.attr("__version__") = SVLLM_VERSION;

  pybind11::class_<svllm::Tokenizer>(m, "Tokenizer")
      .def(pybind11::init<const std::filesystem::path&>(), pybind11::arg("tokenizer_json"))
      .def("encode", &svllm::Tokenizer::encode, pybind11::arg("text"))
      .def(
          "decode",
          [](svllm::Tokenizer& tokenizer, const std::vector<int32_t>& token_ids) {
            return tokenizer.decode(token_ids);
          },
          pybind11::arg("token_ids"));
}
