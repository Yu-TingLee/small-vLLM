#!/usr/bin/env bash
# Prints the toolchain a GPU result depends on and checks it against docs/design-spec.md.
# Paste its output into the PR next to GPU test results. Exits non-zero if a check fails.
# Usage: scripts/check_env.sh
set -euo pipefail

# Targets from docs/design-spec.md, and CMake's minimum from CMakeLists.txt.
COMPUTE_CAPS="8.6 8.9"
MIN_DRIVER=525
MIN_CUDA=12.6
TORCH_CUDA=12.6
MIN_CMAKE=3.26

cd "$(dirname "$0")/.."
failed=0

report() {
  printf '%-4s  %-8s  %s\n' "$1" "$2" "$3"
  if [[ "$1" == FAIL ]]; then failed=1; fi
}

# Succeeds when dotted version $1 is at least $2, e.g. "580.173.02" vs "525".
version_ge() {
  local -a have want
  local i
  IFS=. read -ra have <<<"$1"
  IFS=. read -ra want <<<"$2"
  for ((i = 0; i < ${#want[@]}; i++)); do
    if ((10#${have[i]:-0} > 10#${want[i]})); then return 0; fi
    if ((10#${have[i]:-0} < 10#${want[i]})); then return 1; fi
  done
}

has_gpu=0
if ! command -v nvidia-smi >/dev/null 2>&1; then
  report SKIP gpu "nvidia-smi not found"
  report SKIP driver "no GPU"
  report SKIP nvcc "no GPU"
elif ! gpus=$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader); then
  report FAIL gpu "nvidia-smi failed; is the NVIDIA driver loaded?"
  report SKIP driver "no GPU"
  report SKIP nvcc "no GPU"
else
  has_gpu=1
  while IFS=, read -r index name cap gpu_driver; do
    name=${name# }
    cap=${cap// /}
    driver=${gpu_driver// /}
    if [[ " $COMPUTE_CAPS " == *" $cap "* ]]; then
      report PASS gpu "$index: $name (sm_${cap/./})"
    else
      report FAIL gpu "$index: $name (sm_${cap/./}); needs compute capability ${COMPUTE_CAPS// / or }"
    fi
  done <<<"$gpus"

  if version_ge "$driver" "$MIN_DRIVER"; then
    report PASS driver "$driver"
  else
    report FAIL driver "$driver; needs $MIN_DRIVER or later"
  fi

  # CMake looks for nvcc the same way: CUDACXX, then PATH.
  nvcc=${CUDACXX:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
  if [[ ! -x "$nvcc" ]]; then
    report FAIL nvcc "not found; set CUDACXX or add the CUDA toolkit to PATH"
  else
    cuda=$("$nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
    if version_ge "$cuda" "$MIN_CUDA"; then
      report PASS nvcc "$cuda ($nvcc)"
    else
      report FAIL nvcc "$cuda ($nvcc); needs $MIN_CUDA or later"
    fi
  fi
fi

if ! torch=$(uv run --no-sync python -c \
  'import torch; print(torch.__version__, torch.version.cuda or "cpu")' 2>/dev/null); then
  report FAIL torch "not importable; run uv sync --extra cu126 (or --extra cpu)"
else
  read -r torch_version torch_cuda <<<"$torch"
  if ((has_gpu)) && [[ "$torch_cuda" != "$TORCH_CUDA" ]]; then
    report FAIL torch "$torch_version, CUDA $torch_cuda; run uv sync --extra cu126"
  else
    report PASS torch "$torch_version, CUDA $torch_cuda"
  fi
fi

cxx=${CXX:-c++}
if cxx_version=$("$cxx" --version 2>/dev/null); then
  report INFO cxx "${cxx_version%%$'\n'*}"
else
  report FAIL cxx "$cxx not found"
fi

if ! cmake_version=$(uv run --no-sync cmake --version 2>/dev/null); then
  report FAIL cmake "not found; run uv sync"
else
  cmake_version=${cmake_version%%$'\n'*}
  cmake_version=${cmake_version##* }
  cmake_version=${cmake_version%%-*}
  if version_ge "$cmake_version" "$MIN_CMAKE"; then
    report PASS cmake "$cmake_version"
  else
    report FAIL cmake "$cmake_version; needs $MIN_CMAKE or later"
  fi
fi

exit "$failed"
