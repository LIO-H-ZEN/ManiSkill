#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/maniskill-lift-uv-cache}"

# A proxy bound to localhost on the devbox is not reachable from a GPU worker.
# Drop only that clearly invalid inherited configuration; keep any explicit
# remote/company proxy supplied by the caller.
for proxy_variable in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
  proxy_value="${!proxy_variable:-}"
  if [[ "${proxy_value}" == http://127.0.0.1:* ]] \
    || [[ "${proxy_value}" == https://127.0.0.1:* ]] \
    || [[ "${proxy_value}" == http://localhost:* ]] \
    || [[ "${proxy_value}" == https://localhost:* ]]; then
    unset "${proxy_variable}"
  fi
done

# Some workers expose an older host driver together with NVIDIA's CUDA 12.6
# forward-compatibility package. Prefer that libcuda when it is available.
CUDA_COMPAT_DIR="/usr/local/cuda-12.6/compat"
if [[ -d "${CUDA_COMPAT_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Keep the multi-gigabyte CUDA environment on the machine/worker's local disk.
# The repository itself may live on a much smaller shared NAS volume.
VENV_DIR="${MANISKILL_LIFT_VENV:-${TMPDIR:-/tmp}/maniskill-lift-venv-${UID}}"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  uv venv --python 3.11 "${VENV_DIR}"
fi

if ! "${PYTHON_BIN}" - 2>/dev/null <<'PY'
from importlib.metadata import version

assert version("numpy") == "1.26.4"
assert version("torch") == "2.7.1"
import gymnasium  # noqa: F401
import mplib  # noqa: F401
import sapien  # noqa: F401
PY
then
  echo "Installing Lift task dependencies into ${VENV_DIR}" >&2
  uv pip install \
    --python "${PYTHON_BIN}" \
    --requirements "${REPO_ROOT}/requirements-lift.txt"
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_piper_lift_tasks.py" "$@"
