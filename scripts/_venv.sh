#!/bin/bash
# The one venv shared by every RSA job and the paper's figure-data jobs
# (subjective randomness, number game, RSA chain models). The holdout sweeps
# keep a venv per WORK_ROOT instead (subjective_randomness/slurm/_env.sh).
#
#   shared_venv_path   $GROUP_HOME/venvs/auto-psych (Sherlock's rule: Python
#                      environments live in $GROUP_HOME, not on $SCRATCH);
#                      UV_PROJECT_ENVIRONMENT, if exported, wins; nothing
#                      without GROUP_HOME. Export it before sourcing _env.sh,
#                      which derives VENV_PY from it.
#   build_shared_venv  after _env.sh: build the venv, or bring an existing one
#                      up to the pins, then check it imports.
#
# VENV_MODE=sync: `uv sync --locked --no-build` -- the lock exactly,
#   failing (not building from source) if a package has no glibc-2.17 wheel.
# VENV_MODE=wheels (default; on Sherlock the locked sync fails: the rsa
#   group's ml-dtypes 0.5.4 and contourpy 1.3.3 have no glibc-2.17 wheel):
#   uv pip install --only-binary with the stack pinned to uv.lock's versions.

shared_venv_path() {
  if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    echo "$UV_PROJECT_ENVIRONMENT"
  elif [[ -n "${GROUP_HOME:-}" ]]; then
    echo "$GROUP_HOME/venvs/auto-psych"
  fi
}

build_shared_venv() {
  echo "[venv] $UV_PROJECT_ENVIRONMENT (python $UV_PYTHON, mode ${VENV_MODE:-wheels})"
  case "${VENV_MODE:-wheels}" in
    sync)
      uv sync --locked --no-build ;;
    wheels)
      [[ -x "$VENV_PY" ]] || uv venv --python "$UV_PYTHON" "$UV_PROJECT_ENVIRONMENT"
      uv pip install --python "$UV_PROJECT_ENVIRONMENT" --only-binary :all: \
        "pymc==5.28.5" "pytensor==2.38.3" "arviz==0.23.4" "h5py<3.15" \
        "memo-lang==1.3.0" "jax==0.7.0" "jaxlib==0.7.0" "numpyro==0.19.0" \
        "numpy<2.3" "scipy<1.15" "pandas<2.3" \
        pyyaml tyro pyprojroot matplotlib requests threadpoolctl \
        "google-genai==2.8.0" "pydantic==2.13.4" "resvg-py==0.5.0" ;;  # the open campaign (src/rsa_open)
    *) echo "ERROR: VENV_MODE must be sync or wheels" >&2; return 1 ;;
  esac
  "$VENV_PY" - <<'PY'
import sys
import jax, jax.numpy as jnp, numpyro, memo, pymc, arviz, pandas  # noqa: F401
import google.genai, pydantic, resvg_py  # noqa: F401  (the open campaign's pilot and paradigm specs)
assert sys.version_info[:2] == (3, 12), sys.version
assert jax.__version__ == "0.7.0", jax.__version__
print(f"[venv] OK: python {sys.version.split()[0]}, jax {jax.__version__}, "
      f"numpyro {numpyro.__version__}, pymc {pymc.__version__}, arviz {arviz.__version__}; "
      f"jnp check {float(jnp.ones(3).sum())}")
PY
}
