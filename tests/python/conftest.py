import os
import sys
from pathlib import Path

import torch  # noqa: F401  # loads libtorch before svllm_ops, which links against it

REPO_ROOT = Path(__file__).resolve().parents[2]

build_dir = Path(os.environ.get("SVLLM_BUILD_DIR", "build/release"))
if not build_dir.is_absolute():
    build_dir = REPO_ROOT / build_dir
sys.path.insert(0, str(build_dir / "python"))
