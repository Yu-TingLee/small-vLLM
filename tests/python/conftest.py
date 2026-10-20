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

import svllm_ops  # noqa: E402

BF16_EPS = torch.finfo(torch.bfloat16).eps


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not (svllm_ops.built_with_cuda and torch.cuda.is_available()),
                reason="needs a CUDA build of svllm_ops and a GPU",
            ),
        ),
    ]
)
def device(request) -> torch.device:
    """Run the test once per device an op has a kernel for (ADR-0002)."""
    return torch.device(request.param)


def assert_close_bf16(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 2 * BF16_EPS,
    atol: float = 2 * BF16_EPS,
) -> None:
    """Assert a bf16 op output matches its PyTorch reference to within two bf16 rounding steps.

    Kernels accumulate in fp32 in a different order than PyTorch, which can tip a value to the
    neighbouring bf16 number before and after a second rounding. Ops whose error grows with the
    reduction length, such as attention, pass a looser tolerance.
    """
    assert actual.dtype == torch.bfloat16, f"expected a bf16 output, got {actual.dtype}"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
