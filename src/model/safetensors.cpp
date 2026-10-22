#include "model/safetensors.h"

#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
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
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace svllm {
namespace {

static_assert(std::endian::native == std::endian::little,
              "safetensors stores little-endian values and big-endian hosts are not supported");

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) : size_(std::filesystem::file_size(path)) {
    TORCH_CHECK(size_ >= sizeof(uint64_t), path.string(), " is truncated: ", size_,
                " bytes, too short for the 8-byte header length");
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    TORCH_CHECK(fd >= 0, "cannot open ", path.string(), ": ", std::strerror(errno));
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    const int mmap_errno = errno;
    ::close(fd);
    TORCH_CHECK(data_ != MAP_FAILED, "cannot mmap ", path.string(), ": ",
                std::strerror(mmap_errno));
  }

  ~MappedFile() { ::munmap(data_, size_); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const std::byte* data() const { return static_cast<const std::byte*>(data_); }
  size_t size() const { return size_; }

 private:
  void* data_ = MAP_FAILED;
  size_t size_ = 0;
};

struct TensorEntry {
  std::string name;
  std::vector<int64_t> shape;
  uint64_t begin;
  uint64_t end;
};

std::optional<std::vector<int64_t>> parse_shape(const nlohmann::json& shape_json) {
  if (!shape_json.is_array() ||
      !std::all_of(shape_json.begin(), shape_json.end(),
                   [](const nlohmann::json& d) { return d.is_number_integer(); })) {
    return std::nullopt;
  }
  auto shape = shape_json.get<std::vector<int64_t>>();
  uint64_t numel = 0;
  if (!std::ranges::all_of(shape, [](int64_t d) { return d >= 0; }) ||
      c10::safe_multiplies_u64(shape, &numel)) {
    return std::nullopt;
  }
  return shape;
}

std::unordered_map<std::string, torch::Tensor> parse_safetensors(const std::filesystem::path& path,
                                                                 c10::Device device) {
  const MappedFile file(path);

  uint64_t header_size = 0;
  std::memcpy(&header_size, file.data(), sizeof(header_size));
  TORCH_CHECK(header_size <= file.size() - sizeof(header_size), "header length ", header_size,
              " in ", path.string(), " exceeds the ", file.size(), "-byte file");
  const auto* header_begin = reinterpret_cast<const char*>(file.data()) + sizeof(header_size);

  const nlohmann::json header = nlohmann::json::parse(header_begin, header_begin + header_size);
  TORCH_CHECK(header.is_object(), "header of ", path.string(), " is not a JSON object");

  // Offsets in the header count from the end of the header, not from the start of the file
  const std::byte* data = file.data() + sizeof(header_size) + header_size;
  const uint64_t data_size = file.size() - sizeof(header_size) - header_size;

  std::vector<TensorEntry> entries;
  for (const auto& [name, info] : header.items()) {
    if (name == "__metadata__") continue;
    const std::string tensor_desc = "tensor '" + name + "' in " + path.string();

    TORCH_CHECK(info.is_object() && info.contains("dtype") && info.at("dtype").is_string() &&
                    info.contains("shape") && info.contains("data_offsets"),
                tensor_desc, " needs a string dtype, a shape and data_offsets");

    const auto dtype_name = info.at("dtype").get<std::string>();
    TORCH_CHECK(dtype_name == "BF16", tensor_desc, " has unsupported dtype ", dtype_name,
                "; expected BF16");

    const std::optional<std::vector<int64_t>> shape = parse_shape(info.at("shape"));
    TORCH_CHECK(shape, tensor_desc, " has invalid shape ", info.at("shape").dump());
    const auto numel = static_cast<uint64_t>(c10::multiply_integers(*shape));

    const nlohmann::json& offsets_json = info.at("data_offsets");
    TORCH_CHECK(offsets_json.is_array() && offsets_json.size() == 2 &&
                    offsets_json[0].is_number_unsigned() && offsets_json[1].is_number_unsigned(),
                tensor_desc, " has data_offsets ", offsets_json.dump(), "; expected [begin, end]");
    const auto [begin, end] = offsets_json.get<std::array<uint64_t, 2>>();
    TORCH_CHECK(begin <= end && end <= data_size, tensor_desc, " has data_offsets [", begin, ", ",
                end, "] outside the ", data_size, "-byte data buffer");
    uint64_t nbytes = 0;
    TORCH_CHECK(!c10::mul_overflows(numel, uint64_t{sizeof(c10::BFloat16)}, &nbytes) &&
                    nbytes == end - begin,
                tensor_desc, " has data_offsets spanning ", end - begin,
                " bytes, but its shape and dtype need ", nbytes);
    const uint64_t file_offset = sizeof(header_size) + header_size + begin;
    TORCH_CHECK(file_offset % sizeof(c10::BFloat16) == 0, tensor_desc, " starts at byte ",
                file_offset, " of the file, which is misaligned for BF16");

    entries.push_back({name, *shape, begin, end});
  }

  // The format requires the tensors to tile the data buffer with no gaps or overlaps
  std::ranges::sort(entries, {},
                    [](const TensorEntry& entry) { return std::pair(entry.begin, entry.end); });
  uint64_t covered = 0;
  for (const TensorEntry& entry : entries) {
    TORCH_CHECK(entry.begin == covered, "tensor '", entry.name, "' in ", path.string(),
                " starts at byte ", entry.begin,
                " of the data buffer, but the previous tensor ends at byte ", covered);
    covered = entry.end;
  }
  TORCH_CHECK(covered == data_size, path.string(), " has ", data_size - covered,
              " bytes after its last tensor");

  std::unordered_map<std::string, torch::Tensor> tensors;
  for (const TensorEntry& entry : entries) {
    // The file is unmapped on return, so copy even when the tensor is already on `device`.
    // const_cast is safe: from_blob wants a mutable pointer, but the copy only reads.
    const torch::Tensor view =
        torch::from_blob(const_cast<std::byte*>(data + entry.begin), entry.shape, torch::kBFloat16);
    tensors.emplace(entry.name, view.to(device, torch::kBFloat16, /*non_blocking=*/false,
                                        /*copy=*/true));
  }
  return tensors;
}

}  // namespace

std::unordered_map<std::string, torch::Tensor> load_safetensors(const std::filesystem::path& path,
                                                                c10::Device device) {
  try {
    return parse_safetensors(path, device);
  } catch (const c10::Error&) {
    throw;
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "cannot load ", path.string(), ": ", e.what());
  }
}

}  // namespace svllm
