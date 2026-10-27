#include "engine/device.h"

#include <c10/util/Exception.h>
#include <gtest/gtest.h>

namespace svllm {
namespace {

TEST(ParseDevice, AcceptsCudaAndCpu) {
  EXPECT_EQ(parse_device("cuda"), c10::Device(c10::kCUDA, 0));
  EXPECT_EQ(parse_device("cpu"), c10::Device(c10::kCPU));
}

TEST(ParseDevice, RejectsUnknownName) {
  EXPECT_THROW(parse_device("tpu"), c10::Error);
}

TEST(Describe, CpuReportsThreadCount) {
  EXPECT_NE(describe(c10::kCPU).find("threads"), std::string::npos);
}

}  // namespace
}  // namespace svllm
