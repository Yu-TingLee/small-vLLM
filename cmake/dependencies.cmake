# Third-party C++ dependencies, each pinned to a release tarball and its SHA256.
include(FetchContent)

FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa)
FetchContent_MakeAvailable(json)

if(SVLLM_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Only tokenizers-cpp's Rust C API (tokenizers_c.h): its C++ layer always encodes without special
# tokens, and its CMake needs git submodules the tarball lacks. rust/ has no CMakeLists.txt, so
# FetchContent only downloads. The crate ships no Cargo.lock; ours pins its Rust dependencies.
FetchContent_Declare(tokenizers_cpp
  URL https://github.com/mlc-ai/tokenizers-cpp/archive/c586c52f93f7b060753bd2388eb96a105cb7374d.tar.gz
  URL_HASH SHA256=257acdeb860e977efc92a6266a53dd94b6d82401d195195df1168296b808ddfe
  SOURCE_SUBDIR rust)
FetchContent_MakeAvailable(tokenizers_cpp)
configure_file(${CMAKE_CURRENT_LIST_DIR}/tokenizers_c.Cargo.lock
               ${tokenizers_cpp_SOURCE_DIR}/rust/Cargo.lock COPYONLY)

find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(tokenizers_c_lib ${tokenizers_cpp_BINARY_DIR}/release/libtokenizers_c.a)
add_custom_command(
  OUTPUT ${tokenizers_c_lib}
  COMMAND ${CARGO_EXECUTABLE} build --release --locked
          --manifest-path ${tokenizers_cpp_SOURCE_DIR}/rust/Cargo.toml
          --target-dir ${tokenizers_cpp_BINARY_DIR}
  DEPENDS ${tokenizers_cpp_SOURCE_DIR}/rust/Cargo.lock
  COMMENT "Building tokenizers-cpp's Rust library")
add_custom_target(tokenizers_c_build DEPENDS ${tokenizers_c_lib})
add_library(tokenizers_c STATIC IMPORTED)
set_target_properties(tokenizers_c PROPERTIES
  IMPORTED_LOCATION ${tokenizers_c_lib}
  INTERFACE_INCLUDE_DIRECTORIES ${tokenizers_cpp_SOURCE_DIR}/include
  INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS}")
add_dependencies(tokenizers_c tokenizers_c_build)
