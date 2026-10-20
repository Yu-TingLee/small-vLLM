#include "model/safetensors.h"

#include <c10/util/Exception.h>
#include <c10/util/safe_numerics.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

namespace svllm {
namespace {

static_assert(std::endian::native == std::endian::little,
              "safetensors stores little-endian values and big-endian hosts are not supported");

// A whole file mapped copy-on-write, so in-place ops on zero-copy tensors never reach the disk.
class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) : size_(std::filesystem::file_size(path)) {
    TORCH_CHECK(size_ >= sizeof(uint64_t), path.string(), " is truncated: ", size_,
                " bytes, too short for the 8-byte header length");
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    TORCH_CHECK(fd >= 0, "cannot open ", path.string(), ": ", std::strerror(errno));
    data_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    const int mmap_errno = errno;
    ::close(fd);
    TORCH_CHECK(data_ != MAP_FAILED, "cannot mmap ", path.string(), ": ",
                std::strerror(mmap_errno));
  }

  ~MappedFile() { ::munmap(data_, size_); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  std::byte* data() const { return static_cast<std::byte*>(data_); }
  size_t size() const { return size_; }

 private:
  void* data_ = MAP_FAILED;
  size_t size_ = 0;
};

std::optional<torch::ScalarType> parse_dtype(const std::string& name) {
  if (name == "BF16") return torch::kBFloat16;
  if (name == "F16") return torch::kHalf;
  if (name == "F32") return torch::kFloat;
  if (name == "F64") return torch::kDouble;
  return std::nullopt;
}

std::unordered_map<std::string, torch::Tensor> parse_safetensors(const std::filesystem::path& path,
                                                                 Device device) {
  const auto file = std::make_shared<const MappedFile>(path);

  uint64_t header_size = 0;
  std::memcpy(&header_size, file->data(), sizeof(header_size));
  TORCH_CHECK(header_size <= file->size() - sizeof(header_size), "header length ", header_size,
              " in ", path.string(), " exceeds the ", file->size(), "-byte file");
  const auto* header_begin = reinterpret_cast<const char*>(file->data()) + sizeof(header_size);
  const nlohmann::json header = nlohmann::json::parse(header_begin, header_begin + header_size);
  TORCH_CHECK(header.is_object(), "header of ", path.string(), " is not a JSON object");

  // Offsets in the header count from the end of the header, not from the start of the file.
  std::byte* data = file->data() + sizeof(header_size) + header_size;
  const uint64_t data_size = file->size() - sizeof(header_size) - header_size;
  const torch::Device target = device == Device::kCUDA ? torch::kCUDA : torch::kCPU;

  std::unordered_map<std::string, torch::Tensor> tensors;
  for (const auto& [name, info] : header.items()) {
    if (name == "__metadata__") continue;

    const auto dtype_name = info.at("dtype").get<std::string>();
    const std::optional<torch::ScalarType> dtype = parse_dtype(dtype_name);
    TORCH_CHECK(dtype, "tensor '", name, "' in ", path.string(), " has unsupported dtype ",
                dtype_name, "; expected BF16, F16, F32 or F64");

    const auto shape = info.at("shape").get<std::vector<int64_t>>();
    uint64_t numel = 0;
    TORCH_CHECK(std::ranges::all_of(shape, [](int64_t d) { return d >= 0; }) &&
                    !c10::safe_multiplies_u64(shape, &numel),
                "tensor '", name, "' in ", path.string(), " has invalid shape ",
                info.at("shape").dump());

    const auto [begin, end] = info.at("data_offsets").get<std::array<uint64_t, 2>>();
    TORCH_CHECK(begin <= end && end <= data_size, "tensor '", name, "' in ", path.string(),
                " has data_offsets [", begin, ", ", end, "] outside the ", data_size,
                "-byte data buffer");
    uint64_t nbytes = 0;
    TORCH_CHECK(!c10::mul_overflows(numel, uint64_t{c10::elementSize(*dtype)}, &nbytes) &&
                    nbytes == end - begin,
                "tensor '", name, "' in ", path.string(), " has data_offsets spanning ",
                end - begin, " bytes, but its shape and dtype need ", nbytes);

    // The deleter owns a reference to the mapping, so it outlives every tensor that views it.
    const torch::Tensor view = torch::from_blob(
        data + begin, shape, [file](void*) {}, torch::TensorOptions().dtype(*dtype));
    tensors.emplace(name, view.to(target, torch::kBFloat16));
  }
  return tensors;
}

}  // namespace

std::unordered_map<std::string, torch::Tensor> load_safetensors(const std::filesystem::path& path,
                                                                Device device) {
  try {
    return parse_safetensors(path, device);
  } catch (const c10::Error&) {
    throw;
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "cannot load ", path.string(), ": ", e.what());
  }
}

}  // namespace svllm
