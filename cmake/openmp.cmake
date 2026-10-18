# svllm_openmp: OpenMP for the CPU kernels.
add_library(svllm_openmp INTERFACE)

if(APPLE)
  # AppleClang ships no OpenMP, so use the libomp bundled with torch: a second runtime in the
  # same process aborts. Its install name is a path on torch's build machine, so link against a
  # copy renamed to @rpath/libomp.dylib; at run time that resolves to torch's lib directory.
  set(svllm_omp_dir "${CMAKE_BINARY_DIR}/libomp")
  file(COPY "${TORCH_INSTALL_PREFIX}/lib/libomp.dylib" DESTINATION "${svllm_omp_dir}")
  execute_process(
    COMMAND install_name_tool -id @rpath/libomp.dylib "${svllm_omp_dir}/libomp.dylib"
    COMMAND_ERROR_IS_FATAL ANY)
  target_compile_options(svllm_openmp INTERFACE -Xpreprocessor -fopenmp)
  # A plain -L keeps the copy's directory out of the rpath.
  target_link_options(svllm_openmp INTERFACE "-L${svllm_omp_dir}" -lomp)
else()
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(svllm_openmp INTERFACE OpenMP::OpenMP_CXX)
endif()
