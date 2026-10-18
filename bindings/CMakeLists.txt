Python_add_library(svllm_ops MODULE WITH_SOABI svllm_ops.cpp)
target_link_libraries(svllm_ops PRIVATE svllm ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(svllm_ops PRIVATE TORCH_EXTENSION_NAME=svllm_ops ${TORCH_PYBIND11_ABI})
set_target_properties(svllm_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
