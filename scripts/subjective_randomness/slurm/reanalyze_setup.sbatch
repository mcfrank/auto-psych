#!/bin/bash
#SBATCH --job-name=reanalyze_setup
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Stage 1/2 of the exhaustive-pool reanalysis: build the venv ONCE under
# $WORK_ROOT (so the array tasks don't race on it). Mirrors holdout_setup.sbatch's
# venv build, but stages no GT snapshot — reanalysis reads the generators from
# the repo, not from a sandboxed copy.
#
# The ONE difference from the run-job venv is that we install h5netcdf + h5py
# explicitly: reanalysis READS each run's `.nc` MCMC cache, and those files are
# HDF5-backed NetCDF4, so the reader needs a working HDF5 backend (h5netcdf with
# its h5py runtime). Without it az.from_netcdf would fail and every model would
# be re-sampled from scratch instead of hitting the cache.

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

echo "[setup] building venv at $UV_PROJECT_ENVIRONMENT (python $UV_PYTHON)"
# Reuse only a venv that already imports the HDF5 reader; otherwise rebuild.
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import numpy, pymc, h5netcdf, h5py" >/dev/null 2>&1; then
  echo "[setup] reusing existing venv"
else
  uv venv --clear --python "$UV_PYTHON" "$UV_PROJECT_ENVIRONMENT"
fi

# Only-binary: never build from source on el7. Pins match holdout_setup.sbatch
# (the env that wrote the caches), plus the HDF5 NetCDF backend for reading them.
uv pip install --python "$UV_PROJECT_ENVIRONMENT" --only-binary :all: \
  "pymc==5.28.5" "pytensor==2.38.3" "arviz<1" \
  numpy scipy pyyaml tyro pyprojroot matplotlib \
  h5netcdf h5py

# Fail fast: syntax-check the reanalysis CLI and import the chain it needs, so a
# missing dependency aborts here (afterok) rather than after the array launches.
"$VENV_PY" -m py_compile \
  scripts/subjective_randomness/reanalyze_holdout_exhaustive.py \
  src/subjective_randomness/holdout_recovery.py
"$VENV_PY" - <<'PY'
import sys
from pyprojroot import here
sys.path.insert(0, str(here()))
import arviz as az                                       # noqa: F401
import src.subjective_randomness.holdout_recovery        # noqa: F401
import src.pipelines.outer_loop.orchestrator             # noqa: F401
# A no-group InferenceData round-trips through whatever default engine arviz
# picks; force the HDF5 path so a broken backend fails the setup, not 80 tasks.
import xarray as xr
import numpy as np
import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "probe.nc"
    xr.Dataset({"x": ("i", np.arange(3.0))}).to_netcdf(p, engine="h5netcdf")
    xr.open_dataset(p, engine="h5netcdf").close()
print("[setup] import chain + h5netcdf round-trip OK")
PY

echo "[setup] done."
