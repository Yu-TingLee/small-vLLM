import os
import sys
from pathlib import Path

import pytest
import torch  # loads libtorch before svllm_ops, which links against it

REPO_ROOT = Path(__file__).resolve().parents[2]

build_dir = Path(os.environ.get("SVLLM_BUILD_DIR", "build/release"))
if not build_dir.is_absolute():
    build_dir = REPO_ROOT / build_dir
sys.path.insert(0, str(build_dir / "python"))

BF16_EPS = torch.finfo(torch.bfloat16).eps


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    """Run the test once per device an op has a kernel for (docs/design-spec.md)."""
    if request.param == "cuda":
        import svllm_ops

        if not (svllm_ops.built_with_cuda and torch.cuda.is_available()):
            pytest.skip("needs a CUDA build of svllm_ops and a GPU")
    return torch.device(request.param)


def random_bf16(*shape: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Return seeded standard-normal bf16 values, identical on every device for a given seed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=torch.bfloat16)


def assert_close_bf16(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 2 * BF16_EPS,
    atol: float = 1e-5,
) -> None:
    """Assert a bf16 op output matches its PyTorch reference to within two bf16 rounding steps.

    Kernels accumulate in fp32 in a different order than PyTorch, which can tip a value to the
    neighbouring bf16 number before and after a second rounding. The tolerance is relative, so
    atol only absorbs values that round to about zero. Ops whose error grows with the reduction
    length, such as attention, pass a looser tolerance.
    """
    assert actual.dtype == torch.bfloat16, f"expected a bf16 output, got {actual.dtype}"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
