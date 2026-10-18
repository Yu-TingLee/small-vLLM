add_library(svllm STATIC
  engine/device.cpp
)
target_include_directories(svllm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(svllm PUBLIC
  SVLLM_VERSION="${PROJECT_VERSION}"
  $<$<BOOL:${SVLLM_CUDA}>:SVLLM_CUDA>)
target_link_libraries(svllm PUBLIC ${TORCH_LIBRARIES} svllm_openmp)

add_executable(svllm-bench apps/bench_main.cpp)
target_link_libraries(svllm-bench PRIVATE svllm)
