import json
import os
import urllib.request

tokenizers = {
    "Llama-3.2-1B-Instruct": (
        "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json",
        "tests/cpp/data/llama-3.2-1b-instruct/tokenizer.json",
    ),
    "Qwen3-0.6b": (
        "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json",
        "tests/cpp/data/qwen3-0.6b/tokenizer.json",
    ),
}

for model_name, (url, out_path) in tokenizers.items():
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"

    print(f"Downloading {model_name} tokenizer...")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "small-vLLM-tokenizer-downloader"},
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()

        # Detect a Git LFS pointer instead of the actual JSON file.
        if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                "Server returned a Git LFS pointer, not the actual tokenizer JSON."
            )

        # Verify the downloaded content is valid JSON.
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Tokenizer JSON root is not an object.")

        with open(tmp_path, "wb") as f:
            f.write(content)

        os.replace(tmp_path, out_path)

        print(f"Saved {out_path} ({len(content):,} bytes)")
        print(f"JSON keys: {list(data.keys())[:10]}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"ERROR downloading {model_name}: {e}")
        raise