#include <c10/util/Exception.h>

#include <cstring>
#include <iostream>

#include "engine/device.h"

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
      std::cout << "svllm-bench " << SVLLM_VERSION << "\n";
      return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--device") == 0) {
      std::cout << svllm::describe(svllm::parse_device(argv[2])) << "\n";
      return 0;
    }
  } catch (const c10::Error& e) {
    std::cerr << "error: " << e.what_without_backtrace() << "\n";
    return 1;
  }
  std::cerr << "usage: svllm-bench --version | --device cuda|cpu\n";
  return 2;
}
