cmake_minimum_required(VERSION 3.26)
project(small-vllm VERSION 0.1.0 LANGUAGES CXX)

option(SVLLM_CUDA "Build the CUDA backend" ON)
option(SVLLM_BUILD_TESTS "Build the C++ tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR "GCC ${CMAKE_CXX_COMPILER_VERSION} is too old; torch's headers need GCC 11 or "
                      "newer. Set CXX to a newer g++ and reconfigure with --fresh.")
endif()

if(SVLLM_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 89)  # RTX 3090, RTX 4090
  endif()
  # nvcc otherwise compiles host code with the default c++ on PATH, which may be older than CXX.
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6)
    message(FATAL_ERROR "nvcc ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER}) is too old; "
                        "torch needs CUDA 12.6 or newer. Set CUDACXX to a newer nvcc and "
                        "reconfigure with --fresh.")
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Torch's CMake config reads its arch list from here; without it, it probes the local GPU.
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "^([0-9]+)([0-9])$" "\\1.\\2"
       OUTPUT_VARIABLE torch_cuda_arch_list)
  set(ENV{TORCH_CUDA_ARCH_LIST} "${torch_cuda_arch_list}")
endif()

include(cmake/torch.cmake)
include(cmake/openmp.cmake)

# Everything, fetched dependencies included, must use LibTorch's C++ ABI.
separate_arguments(torch_cxx_flags UNIX_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${torch_cxx_flags})

include(cmake/dependencies.cmake)

add_subdirectory(src)
add_subdirectory(bindings)
if(SVLLM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
