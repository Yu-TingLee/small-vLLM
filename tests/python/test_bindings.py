import tomllib

import svllm_ops
from conftest import REPO_ROOT


def test_module_version_matches_project():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert svllm_ops.__version__ == project["version"]
