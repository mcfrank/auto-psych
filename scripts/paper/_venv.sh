#!/bin/bash
# ============================================================================
#  The figure-data jobs' venv, sourced after _env.sh. Not `uv sync --locked`:
#  Sherlock's compute nodes are el7 (glibc 2.17), and the lock's ml-dtypes
#  0.5.4 (the rsa group's jaxlib) has no glibc-2.17 wheel, so the sync built
#  it from source and failed (paper_sr_setup, 2026-10-19). The figure data
#  need only the PyMC stack: binary wheels only, pinned to uv.lock's
#  versions. Idempotent; a venv that cannot import numpy is rebuilt.
# ============================================================================
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import numpy" >/dev/null 2>&1; then
  echo "[paper] reusing venv $UV_PROJECT_ENVIRONMENT"
else
  uv venv --clear --python "$UV_PYTHON" "$UV_PROJECT_ENVIRONMENT"
fi
uv pip install --python "$UV_PROJECT_ENVIRONMENT" --only-binary :all: \
  "pymc==5.28.5" "pytensor==2.38.3" "arviz==0.23.4" "h5py==3.14.0" \
  "numpy==2.2.6" "scipy==1.14.1" "pandas==2.2.3" \
  "pyyaml==6.0.3" "tyro==1.0.13" "pyprojroot==0.3.0"
