import pytest
import svllm_ops
import torch
import torch.nn.functional as F
from conftest import assert_close_bf16

EPS = 1e-6

SHAPES = [
    (0, 1024),  # an empty batch: no kernel launch
    (1, 128),  # one token, Qwen3 head_dim for QK-norm: fewer vectors than one warp
    (7, 1024),  # Qwen3-0.6B hidden size
    (2, 5, 2048),  # Llama 3.2 1B hidden size, batch and sequence dims flattened into rows
    (3, 4100),  # not a multiple of 8: the one-element-at-a-time path, several per thread
    (2, 16384),  # each thread loads two 16-byte vectors
]


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Definition: RMSNorm as Hugging Face's Qwen3 and Llama compute it, built on torch's rms_norm.
    Formula:    out = bf16(x / sqrt(mean(x^2) + eps)) * weight, normalized in fp32.
    Domain:     bf16 x [..., hidden] and weight [hidden] on one device; eps > 0.
    Returns:    A bf16 tensor shaped like x, the expected output of svllm_ops.rms_norm.
    """
    normed = F.rms_norm(x.float(), (x.shape[-1],), eps=eps).to(torch.bfloat16)
    return normed * weight


def random_bf16(*shape: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=torch.bfloat16)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_matches_torch(device, shape):
    x = random_bf16(*shape, device=device)
    weight = random_bf16(shape[-1], device=device, seed=1)
    assert_close_bf16(svllm_ops.rms_norm(x, weight, EPS), reference(x, weight, EPS))


def test_matches_torch_on_unaligned_rows(device):
    # Starting one element in keeps x contiguous but moves its rows off 16-byte boundaries
    x = random_bf16(4 * 1024 + 1, device=device)[1:].view(4, 1024)
    weight = random_bf16(1024, device=device, seed=1)
    assert_close_bf16(svllm_ops.rms_norm(x, weight, EPS), reference(x, weight, EPS))


@pytest.mark.parametrize(
    "x, weight, message",
    [
        (torch.ones(2, 8), torch.ones(8, dtype=torch.bfloat16), "expected bf16"),
        (torch.ones(2, 8, dtype=torch.bfloat16), torch.ones(4, dtype=torch.bfloat16), "shape"),
        (torch.ones(8, 2, dtype=torch.bfloat16).t(), torch.ones(8, dtype=torch.bfloat16), "contig"),
        (
            torch.ones(2, 8, dtype=torch.bfloat16, device="meta"),
            torch.ones(8, dtype=torch.bfloat16, device="meta"),
            "unsupported device",
        ),
    ],
    ids=["fp32", "weight-mismatch", "non-contiguous", "meta-device"],
)
def test_rejects_invalid_inputs(x, weight, message):
    with pytest.raises(RuntimeError, match=message):
        svllm_ops.rms_norm(x, weight, EPS)
