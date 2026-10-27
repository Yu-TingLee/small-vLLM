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


def test_decode_rejects_ids_outside_vocab(tokenizer_path):
    vocab_size = Tokenizer.from_file(tokenizer_path).get_vocab_size(with_added_tokens=True)
    tok = svllm_ops.Tokenizer(tokenizer_path)

    for bad_id in (-1, vocab_size):
        with pytest.raises(RuntimeError, match="outside the vocabulary"):
            tok.decode([bad_id])
