# LibTorch comes from the pip `torch` package

The engine, the pybind11 test bindings and the PyTorch/Hugging Face references all link against the LibTorch bundled in the pinned pip `torch` wheel (found via `torch.utils.cmake_prefix_path`), never a separately downloaded libtorch zip. A second LibTorch risks version and C++ ABI mismatches between the engine and the test module, which surface as import-time crashes or link errors. C++ and CUDA code use the C++ standard LibTorch's headers require: C++20 as of torch 2.14.
