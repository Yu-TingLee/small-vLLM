import os
import re

import pytest
import svllm_ops
from conftest import REPO_ROOT
from huggingface_hub import get_token, hf_hub_download
from tokenizers import Tokenizer

# The revisions scripts/download_models.sh pins, e.g. {"QWEN3": "c1899de..."}.
REVISIONS = dict(
    re.findall(
        r"^(\w+)_REVISION=(\w+)$",
        (REPO_ROOT / "scripts" / "download_models.sh").read_text(),
        re.MULTILINE,
    )
)

# repo, revision, gated
MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", REVISIONS["QWEN3"], False),
    "llama-3.2-1b-instruct": ("meta-llama/Llama-3.2-1B-Instruct", REVISIONS["LLAMA"], True),
}

TEXTS = [
    "",
    "Hello, world!",
    "  leading spaces,\ttabs\nand newlines  ",
    "你好，世界",
    "naïve café 🦙",
    # Special tokens of Qwen3 and Llama: encode keeps them as one id, decode drops them.
    "<|im_start|>user\nhi<|im_end|>",
    "<|begin_of_text|>hi<|eot_id|>",
]


@pytest.fixture(scope="module", params=MODELS)
def tokenizer_path(request):
    repo, revision, gated = MODELS[request.param]
    if gated and get_token() is None:
        # CI sets this outside fork PRs, which get no secrets, so a lost token can't pass silently.
        if os.environ.get("SVLLM_REQUIRE_GATED") == "true":
            pytest.fail(f"{repo} is gated and HF_TOKEN is unset")
        pytest.skip(f"{repo} is gated; set HF_TOKEN to an account with access")
    return hf_hub_download(repo, "tokenizer.json", revision=revision)


@pytest.mark.parametrize("text", TEXTS)
def test_matches_hugging_face(tokenizer_path, text):
    ref = Tokenizer.from_file(tokenizer_path)
    tok = svllm_ops.Tokenizer(tokenizer_path)

    ids = tok.encode(text)

    assert ids == ref.encode(text).ids
    assert tok.decode(ids) == ref.decode(ids)


# A stray byte, a truncated character, an overlong NUL and a UTF-16 surrogate.
@pytest.mark.parametrize("text", [b"\xff", b"\xe4\xbd", b"\xc0\x80", b"\xed\xa0\x80"])
def test_encode_rejects_invalid_utf8(tokenizer_path, text):
    tok = svllm_ops.Tokenizer(tokenizer_path)

    with pytest.raises(RuntimeError, match="not valid UTF-8"):
        tok.encode(text)


# Qwen3's model has more outputs than its tokenizer has tokens, so sampling can yield such ids.
def test_decode_drops_ids_past_vocab(tokenizer_path):
    ref = Tokenizer.from_file(tokenizer_path)
    vocab_size = ref.get_vocab_size(with_added_tokens=True)
    tok = svllm_ops.Tokenizer(tokenizer_path)
    ids = ref.encode("Hello").ids + [vocab_size, 2**31 - 1]

    assert tok.decode(ids) == ref.decode(ids)


def test_decode_rejects_negative_ids(tokenizer_path):
    tok = svllm_ops.Tokenizer(tokenizer_path)

    with pytest.raises(RuntimeError, match="is negative"):
        tok.decode([-1])
